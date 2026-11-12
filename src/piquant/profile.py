"""Measured latency of each pi0.5 stage on this GPU, timed with CUDA events, next to its roofline floor."""

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from rich.console import Console
from rich.table import Table

from piquant.roofline import FORMATS, GPUS, PI05_ACTION_EXPERT, TF32, Workload, latency_floor, stage_costs
from piquant.timing import CallTimes, Prefix, Stage, Summary, measure, summarize

POLICY = "lerobot/pi05_libero_finetuned"
# LeRobot's pi0.5 sets "high" (TF32) when it compiles, so eager uses it too to time the same math as lerobot-eval.
MATMUL_PRECISION = "high"


@dataclass(frozen=True)
class _Run:
    device: str
    compiled: bool
    steps: int
    prefix: Prefix
    calls: list[CallTimes]

    @property
    def mode(self) -> str:
        return "torch.compile" if self.compiled else "eager"


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("piquant.profile needs a CUDA GPU, and none is available.")
    console = Console()
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    policy = _load_policy(args.policy, args.steps, args.compile)
    config = policy.config
    batch = _synthetic_batch(config)

    warmup = f"Warming up for {args.warmup} calls"
    if args.compile:
        warmup += f"; torch.compile in {config.compile_mode} mode can take minutes"
    console.print(f"{warmup}.")
    torch.cuda.reset_peak_memory_stats()
    calls, prefix = measure(policy, batch, args.warmup, args.iters, breakdown=not args.compile)
    run = _Run(
        device=torch.cuda.get_device_name(),
        compiled=args.compile,
        steps=config.num_inference_steps,
        prefix=prefix or Prefix.from_config(config),
        calls=calls,
    )

    gpu_name = args.gpu or _roofline_gpu(run.device)
    floor = _floors(run.prefix, config.chunk_size, gpu_name) if gpu_name else None
    _print_report(console, run, summarize(calls), floor, gpu_name, _stage_weights(policy.model))

    if args.output:
        _write_record(args.output, args.policy, run, floor)
        console.print(f"Wrote {args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=POLICY)
    parser.add_argument("--steps", type=int, help="denoising steps (default: the checkpoint's)")
    parser.add_argument("--warmup", type=_positive_int, default=3)
    parser.add_argument("--iters", type=_positive_int, default=20)
    parser.add_argument("--compile", action="store_true", help="keep the checkpoint's torch.compile; total only")
    parser.add_argument("--gpu", choices=sorted(GPUS), help="roofline GPU to compare against (default: detected)")
    parser.add_argument("--output", type=Path, help="write every call's stage timings as JSON")
    return parser.parse_args()


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _load_policy(path: str, steps: int | None, compile_model: bool) -> PI05Policy:
    config = PreTrainedConfig.from_pretrained(path)
    config.compile_model = compile_model  # the checkpoint enables torch.compile; its compiled graph hides the stages
    config.device = "cuda"
    if steps is not None:
        config.num_inference_steps = steps
    return PI05Policy.from_pretrained(path, config=config).eval()


def _synthetic_batch(config: PI05Config) -> dict[str, torch.Tensor]:
    """One observation shaped like a LIBERO rollout's. Latency does not depend on the pixel or token values."""
    # The env leaves out the empty cameras, and the policy fills them in with blank images, as in a rollout.
    batch = {
        key: torch.rand(1, *feature.shape, device="cuda")
        for key, feature in config.image_features.items()
        if ".empty_camera_" not in key
    }
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 1000, (1, config.tokenizer_max_length), device="cuda")
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, config.tokenizer_max_length, dtype=torch.bool, device="cuda")
    return batch


def _roofline_gpu(device_name: str) -> str | None:
    name = device_name.removeprefix("NVIDIA ").removeprefix("GeForce ")
    return name if name in GPUS else None


def _floors(prefix: Prefix, action_tokens: int, gpu_name: str) -> dict[str, float]:
    """Each stage's roofline floor in ms, the action expert's for one step."""
    stages = stage_costs(Workload(prefix.image_slots, prefix.text_tokens, action_tokens), PI05_ACTION_EXPERT)
    gpu, bf16 = GPUS[gpu_name], FORMATS["BF16"]
    return {
        # LeRobot keeps SigLIP and the projector in FP32, which MATMUL_PRECISION runs as TF32.
        Stage.VISION: latency_floor(stages.vision_encoder, gpu, TF32).ms,
        Stage.LLM: latency_floor(stages.llm, gpu, bf16).ms,
        Stage.EXPERT: latency_floor(stages.action_expert, gpu, bf16).ms,
    }


def _row_floors(floor: dict[str, float], steps: int) -> dict[str, float]:
    """floors spread over the report's rows. Denoising is the expert's floor once per step, and the expert row
    sums its passes over the call the same way."""
    denoising = steps * floor[Stage.EXPERT]
    return {
        Stage.VISION: floor[Stage.VISION],
        Stage.LLM: floor[Stage.LLM],
        Stage.DENOISING: denoising,
        Stage.EXPERT: denoising,
        Stage.STEP: floor[Stage.EXPERT],
        Stage.TOTAL: floor[Stage.VISION] + floor[Stage.LLM] + denoising,
    }


def _stage_weights(model: PI05Pytorch) -> dict[str, dict[torch.dtype, int]]:
    """Weight bytes each stage reads per call, by dtype, the action expert's per denoising step."""
    pwe = model.paligemma_with_expert
    vision = _weight_bytes(pwe.paligemma.model.vision_tower)
    for dtype, n in _weight_bytes(pwe.paligemma.model.multi_modal_projector).items():
        vision[dtype] = vision.get(dtype, 0) + n

    def read_per_call(name: str) -> bool:
        # A call looks up a few rows of the token embedding table, never all of it.
        return "embed_tokens" not in name

    # .model leaves out the expert's lm_head, which inference never uses.
    expert = pwe.gemma_expert.model
    return {
        Stage.VISION: vision,
        Stage.LLM: _weight_bytes(pwe.paligemma.model.language_model, select=read_per_call),
        Stage.EXPERT: _weight_bytes(expert, select=read_per_call),
        # The adaRMS modulation layers: they read the timestep only, so their output is the same for every token.
        "of which adaRMS": _weight_bytes(expert, select=lambda name: "norm.dense" in name),
    }


def _weight_bytes(module: torch.nn.Module, select: Callable[[str], bool] = lambda name: True) -> dict[torch.dtype, int]:
    """Bytes of parameters per dtype: what a batch-1 forward pass reads from memory for its weights."""
    total = defaultdict(int)
    for name, param in module.named_parameters():
        if select(name):
            total[param.dtype] += param.numel() * param.element_size()
    return dict(total)


def _print_report(
    console: Console,
    run: _Run,
    summary: dict[str, Summary],
    floor: dict[str, float] | None,
    gpu_name: str | None,
    weights: dict[str, dict[torch.dtype, int]],
) -> None:
    prefix = run.prefix
    console.print(
        f"Prefix: {prefix.tokens} tokens = {prefix.image_slots} images x {prefix.tokens_per_image} "
        f"({prefix.blank_image_slots} blank) + {prefix.text_tokens} text slots."
    )
    if not run.compiled:
        console.print("Vision encoder = SigLIP on every image. Other = preprocessing, masks and Euler updates.")

    title = (
        f"{run.device}, {run.steps} denoising steps, {run.mode}, {MATMUL_PRECISION} matmul, "
        f"per call over {len(run.calls)} calls (ms)"
    )
    if floor:
        title += f". Floor = BF16 roofline on {gpu_name}, TF32 for the vision encoder"
    console.print(_latency_table(title, summary, _row_floors(floor, run.steps) if floor else None, run.steps))
    console.print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    console.print(_weights_table(weights))


def _latency_table(title: str, summary: dict[str, Summary], floor: dict[str, float] | None, steps: int) -> Table:
    labels = {
        Stage.VISION: "Vision encoder",
        Stage.LLM: "LLM prefix",
        Stage.DENOISING: f"Denoising ({steps} steps)",
        Stage.EXPERT: "  action expert",
        Stage.STEP: "  one step",
        Stage.OTHER: "Other",
        Stage.TOTAL: "Total",
    }
    table = Table(title=title)
    table.add_column("Stage")
    for column in ["Median", "p90", "Share", *(["Floor", "x floor"] if floor else [])]:
        table.add_column(column, justify="right")

    total = summary[Stage.TOTAL].median_ms
    for stage in Stage:
        if stage not in summary:
            continue
        s = summary[stage]
        # A step is a slice of denoising, not a share of its own.
        share = "" if stage == Stage.STEP else f"{s.median_ms / total:.0%}"
        row = [labels[stage], f"{s.median_ms:.2f}", f"{s.p90_ms:.2f}", share]
        if floor:
            f = floor.get(stage)
            row += [f"{f:.2f}", f"{s.median_ms / f:.1f}x"] if f else ["", ""]
        table.add_row(*row)
    return table


def _weights_table(weights: dict[str, dict[torch.dtype, int]]) -> Table:
    table = Table(title="Weights each stage reads by dtype (MB), the action expert's once per denoising step")
    table.add_column("Stage")
    table.add_column("Weights by dtype", justify="right")
    for stage, by_dtype in weights.items():
        cells = ", ".join(f"{n / 1e6:.0f} {_dtype_label(d)}" for d, n in sorted(by_dtype.items(), key=lambda x: -x[1]))
        table.add_row(stage, cells)
    return table


def _dtype_label(dtype: torch.dtype) -> str:
    return {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, str(dtype))


def _write_record(path: Path, policy: str, run: _Run, floor: dict[str, float] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "device": run.device,
        "policy": policy,
        "mode": run.mode,
        "denoising_steps": run.steps,
        "torch": torch.__version__,
        "prefix": asdict(run.prefix),
        "floors_ms": floor or {},
        "calls_ms": run.calls,
    }
    path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
