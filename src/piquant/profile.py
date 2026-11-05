"""Measured latency of each pi0.5 stage on this GPU, timed with CUDA events, next to its roofline floor."""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from piquant.roofline import FORMATS, GPUS, PI05_ACTION_EXPERT, TOKENS_PER_IMAGE, Workload, latency_floor, stage_costs

POLICY = "lerobot/pi05_libero_finetuned"
TIMED_STAGES = ("vision encoder", "llm prefix", "denoising", "action expert")
# LeRobot's pi0.5 sets "high" (TF32) when it compiles, so eager uses it too to time the same math as lerobot-eval.
MATMUL_PRECISION = "high"


@dataclass(frozen=True)
class Prefix:
    image_slots: int
    blank_image_slots: int  # empty cameras: masked out of attention but still encoded and run through the LLM
    tokens_per_image: int
    text_tokens: int  # padded to tokenizer_max_length, so all of them are computed

    @property
    def tokens(self) -> int:
        return self.image_slots * self.tokens_per_image + self.text_tokens


@dataclass(frozen=True)
class Summary:
    median_ms: float
    p90_ms: float


def prefix_composition(images, img_masks, tokens, prefix_embs) -> Prefix:
    tokens_per_image = (prefix_embs.shape[1] - tokens.shape[1]) // len(images)
    blank = sum(1 for mask in img_masks if not bool(mask.any()))
    return Prefix(len(images), blank, tokens_per_image, tokens.shape[1])


class StageTimer:
    """Brackets wrapped methods with CUDA events. Events are read after each call, so timing adds no syncs."""

    def __init__(self):
        self._events = defaultdict(list)

    def wrap(self, obj, method: str, stage_of) -> None:
        """Time obj.method under the stage stage_of(*args, **kwargs) returns; None leaves that call untimed."""
        original = getattr(obj, method)

        def timed(*args, **kwargs):
            stage = stage_of(*args, **kwargs)
            if stage is None:
                return original(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self._events[stage].append((start, end))
            return result

        setattr(obj, method, timed)

    def read(self) -> dict[str, list[float]]:
        """Each timed call's ms, per stage, since the last read."""
        torch.cuda.synchronize()
        ms = {stage: [start.elapsed_time(end) for start, end in pairs] for stage, pairs in self._events.items()}
        self._events.clear()
        return ms


def summarize(calls: list[dict[str, list[float]]]) -> dict[str, Summary]:
    """Median and p90 across calls of each stage's time per call (summed over e.g. its denoising steps).

    "other" is whatever the total spends outside the timed stages.
    """
    calls = [{stage: sum(ms) for stage, ms in c.items()} for c in calls]
    for c in calls:
        if "vision encoder" in c:
            c["other"] = c["total"] - c["vision encoder"] - c["llm prefix"] - c["denoising"]
    return {stage: describe([c[stage] for c in calls]) for stage in calls[0]}


def describe(values: list[float]) -> Summary:
    # inclusive keeps p90 within the samples; the default exclusive method extrapolates past the max on few calls.
    p90 = statistics.quantiles(values, n=10, method="inclusive")[-1] if len(values) > 1 else values[0]
    return Summary(statistics.median(values), p90)


def weight_bytes(module: torch.nn.Module, select=lambda name: True) -> dict[torch.dtype, int]:
    """Bytes of parameters per dtype: what a batch-1 forward pass reads from memory for its weights."""
    total = defaultdict(int)
    for name, param in module.named_parameters():
        if select(name):
            total[param.dtype] += param.numel() * param.element_size()
    return dict(total)


def roofline_gpu(device_name: str) -> str | None:
    name = device_name.removeprefix("NVIDIA ").removeprefix("GeForce ")
    return name if name in GPUS else None


def load_policy(path: str, steps: int | None, compile_model: bool):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    config = PreTrainedConfig.from_pretrained(path)
    config.compile_model = compile_model  # the checkpoint enables torch.compile; its compiled graph hides the stages
    config.device = "cuda"
    if steps is not None:
        config.num_inference_steps = steps
    return PI05Policy.from_pretrained(path, config=config).eval()


def synthetic_batch(config) -> dict[str, torch.Tensor]:
    """One observation shaped like a LIBERO rollout's. Latency does not depend on the pixel or token values."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    # The env leaves out the empty cameras, and the policy fills them in with blank images, as in a rollout.
    batch = {
        key: torch.rand(1, *feature.shape, device="cuda")
        for key, feature in config.image_features.items()
        if ".empty_camera_" not in key
    }
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 1000, (1, config.tokenizer_max_length), device="cuda")
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, config.tokenizer_max_length, dtype=torch.bool, device="cuda")
    return batch


def measure(
    policy, batch, warmup: int, iters: int, breakdown: bool
) -> tuple[list[dict[str, list[float]]], Prefix | None]:
    model = policy.model
    timer = StageTimer()
    prefix_inputs = None

    if breakdown:
        embed_prefix = model.embed_prefix

        def capture(images, img_masks, tokens, masks):
            nonlocal prefix_inputs
            out = embed_prefix(images, img_masks, tokens, masks)
            # Keep the tensors, not their values: reading a mask here would sync inside the timed vision encoder.
            prefix_inputs = (images, img_masks, tokens, out[0])
            return out

        model.embed_prefix = capture

        def forward_stage(*args, inputs_embeds, **kwargs):
            prefix_embs, suffix_embs = inputs_embeds
            return "llm prefix" if suffix_embs is None else "action expert" if prefix_embs is None else None

        # embed_prefix runs SigLIP on every image slot; the text embedding lookup it also does is negligible.
        timer.wrap(model, "embed_prefix", lambda *a, **k: "vision encoder")
        timer.wrap(model.paligemma_with_expert, "forward", forward_stage)
        timer.wrap(model, "denoise_step", lambda *a, **k: "denoising")
    timer.wrap(policy, "predict_action_chunk", lambda *a, **k: "total")

    calls = []
    for i in range(warmup + iters):
        policy.predict_action_chunk(batch)
        ms = timer.read()
        if i >= warmup:
            calls.append(ms)
    return calls, prefix_composition(*prefix_inputs) if prefix_inputs else None


def stage_weights(model) -> dict[str, dict[torch.dtype, int]]:
    pwe = model.paligemma_with_expert
    vision = weight_bytes(pwe.paligemma.model.vision_tower)
    for dtype, n in weight_bytes(pwe.paligemma.model.multi_modal_projector).items():
        vision[dtype] = vision.get(dtype, 0) + n

    def read_per_call(name: str) -> bool:
        # A call looks up a few rows of the token embedding table, never all of it.
        return "embed_tokens" not in name

    # .model leaves out the expert's lm_head, which inference never uses.
    expert = pwe.gemma_expert.model
    return {
        "vision encoder": vision,
        "llm prefix": weight_bytes(pwe.paligemma.model.language_model, select=read_per_call),
        "action expert": weight_bytes(expert, select=read_per_call),
        # The adaRMS modulation layers: they read the timestep only, so their output is the same for every token.
        "of which adaRMS": weight_bytes(expert, select=lambda name: "norm.dense" in name),
    }


def floors(prefix: Prefix, action_tokens: int, gpu_name: str) -> dict[str, float]:
    stages = stage_costs(Workload(prefix.image_slots, prefix.text_tokens, action_tokens), PI05_ACTION_EXPERT)
    gpu, bf16 = GPUS[gpu_name], FORMATS["BF16"]
    return {
        "vision encoder": latency_floor(stages.vision_encoder, gpu, bf16).ms,
        "llm prefix": latency_floor(stages.llm, gpu, bf16).ms,
        "action expert": latency_floor(stages.action_expert, gpu, bf16).ms,
    }


def dtype_label(dtype: torch.dtype) -> str:
    return {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, str(dtype))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=POLICY)
    parser.add_argument("--steps", type=int, help="denoising steps (default: the checkpoint's)")
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--iters", type=positive_int, default=20)
    parser.add_argument("--compile", action="store_true", help="keep the checkpoint's torch.compile; total only")
    parser.add_argument("--gpu", choices=sorted(GPUS), help="roofline GPU to compare against (default: detected)")
    parser.add_argument("--output", type=Path, help="write every call's stage timings as JSON")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.exit(1, "piquant.profile needs a CUDA GPU, and none is available.\n")

    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    policy = load_policy(args.policy, args.steps, args.compile)
    config = policy.config
    steps = config.num_inference_steps
    device = torch.cuda.get_device_name()
    console = Console()
    warmup = f"Warming up for {args.warmup} calls"
    if args.compile:
        warmup += f"; torch.compile in {config.compile_mode} mode can take minutes"
    console.print(f"{warmup}.")
    torch.cuda.reset_peak_memory_stats()
    calls, prefix = measure(policy, synthetic_batch(config), args.warmup, args.iters, breakdown=not args.compile)
    if prefix is None:
        # Compiled runs have no hooks to see the prefix, so read it off the config, as the policy builds it.
        image_slots = len(config.image_features)
        prefix = Prefix(image_slots, config.empty_cameras, TOKENS_PER_IMAGE, config.tokenizer_max_length)
    summary = summarize(calls)
    gpu_name = args.gpu or roofline_gpu(device)
    floor = floors(prefix, config.chunk_size, gpu_name) if gpu_name else {}

    mode = "torch.compile" if args.compile else "eager"
    title = f"{device}, {steps} denoising steps, {mode}, {MATMUL_PRECISION} matmul, per call over {args.iters} calls (ms)"
    if floor:
        title += f". Floor = BF16 roofline on {gpu_name}"
    table = Table(title=title)
    for column in ["Stage", "Median", "p90", "Share"]:
        table.add_column(column, justify="left" if column == "Stage" else "right")
    if floor:
        table.add_column("Floor", justify="right")
        table.add_column("x floor", justify="right")

    total = summary["total"].median_ms
    row_floor = {}
    if floor:
        expert = steps * floor["action expert"]
        row_floor = {
            "vision encoder": floor["vision encoder"],
            "llm prefix": floor["llm prefix"],
            "denoising": expert,
            "action expert": expert,
            "step": floor["action expert"],
            "total": floor["vision encoder"] + floor["llm prefix"] + expert,
        }
    if "denoising" in summary:
        # The typical single step. A step's sum over a call also carries the occasional stall, which eager
        # launch-bound runs show as steps several times slower than the rest.
        summary["step"] = describe([ms for c in calls for ms in c["denoising"]])
    labels = {
        "vision encoder": "Vision encoder",
        "llm prefix": "LLM prefix",
        "denoising": f"Denoising ({steps} steps)",
        "action expert": "  action expert",
        "step": "  one step",
        "other": "Other",
        "total": "Total",
    }
    for stage in [*TIMED_STAGES, "step", "other", "total"]:
        if stage not in summary:
            continue
        s = summary[stage]
        share = "" if stage == "step" else f"{s.median_ms / total:.0%}"
        row = [labels[stage], f"{s.median_ms:.2f}", f"{s.p90_ms:.2f}", share]
        if row_floor:
            f = row_floor.get(stage)
            row += [f"{f:.2f}", f"{s.median_ms / f:.1f}x"] if f else ["", ""]
        table.add_row(*row)

    console.print(
        f"Prefix: {prefix.tokens} tokens = {prefix.image_slots} images x {prefix.tokens_per_image} "
        f"({prefix.blank_image_slots} blank) + {prefix.text_tokens} text slots."
    )
    if not args.compile:
        console.print("Vision encoder = SigLIP on every image. Other = preprocessing, masks and Euler updates.")
    console.print(table)
    console.print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    weights = Table(title="Weights each stage reads by dtype (MB), the action expert's once per denoising step")
    weights.add_column("Stage")
    weights.add_column("Weights by dtype", justify="right")
    for stage, by_dtype in stage_weights(policy.model).items():
        cells = ", ".join(f"{n / 1e6:.0f} {dtype_label(d)}" for d, n in sorted(by_dtype.items(), key=lambda x: -x[1]))
        weights.add_row(stage, cells)
    console.print(weights)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "device": device,
            "policy": args.policy,
            "mode": mode,
            "denoising_steps": steps,
            "torch": torch.__version__,
            "prefix": asdict(prefix),
            "floors_ms": floor,
            "calls_ms": calls,
        }
        args.output.write_text(json.dumps(record, indent=2))
        console.print(f"Wrote {args.output}")


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


if __name__ == "__main__":
    main()
