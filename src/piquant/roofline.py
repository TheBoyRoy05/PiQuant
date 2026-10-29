"""Latency floors for pi0.5 inference: each stage takes at least max(FLOPs / peak, bytes / bandwidth)."""

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class Gpu:
    bf16_flops_per_s: float  # dense tensor-core FLOP/s with FP32 accumulate
    int8_ops_per_s: float
    bandwidth_bytes_per_s: float


@dataclass(frozen=True)
class Format:
    weight_bits: int
    activation_bits: int  # sets the tensor-core math


@dataclass(frozen=True)
class Workload:
    cameras: int  # number of images
    text_tokens: int
    action_tokens: int


@dataclass(frozen=True)
class StageCost:
    linear_flops: float
    attention_flops: float
    weight_params: int
    kv_cache_bytes: float


@dataclass(frozen=True)
class Stages:
    vision_encoder: StageCost
    llm: StageCost
    action_expert: StageCost  # one denoising step


@dataclass(frozen=True)
class LatencyFloor:
    ms: float
    compute_bound: bool


@dataclass(frozen=True)
class Transformer:
    layers: int
    width: int
    mlp_width: int
    heads: int
    kv_heads: int
    head_dim: int
    gated_mlp: bool
    adaptive_norm: bool = False  # pi0.5 conditions the expert's norms on the flow timestep; pi0 does not

    @property
    def kv_width(self) -> int:
        return 2 * self.kv_heads * self.head_dim

    @property
    def kv_params(self) -> int:
        return self.width * self.kv_width

    @property
    def layer_params(self) -> int:
        q_projection = self.width * self.heads * self.head_dim
        output_projection = self.heads * self.head_dim * self.width
        mlp = (3 if self.gated_mlp else 2) * self.width * self.mlp_width
        return q_projection + self.kv_params + output_projection + mlp

    @property
    def token_params(self) -> int:
        return self.layers * self.layer_params

    @property
    def adaptive_norm_params(self) -> int:
        # Two norms per layer, each a dense layer producing scale, shift and gate. They act on one timestep vector,
        # so they add weights to read but negligible FLOPs.
        return self.layers * 2 * self.width * 3 * self.width if self.adaptive_norm else 0

    @property
    def weight_params(self) -> int:
        return self.token_params + self.adaptive_norm_params

    def layer_attention_flops(self, queries: int, keys: int) -> float:
        return 4 * queries * keys * self.heads * self.head_dim  # Q @ K^T and probs @ V


KV_CACHE_BYTES_PER_VALUE = 2  # the prefix KV cache stays BF16 in every format
DENOISING_STEPS = (10, 2)  # pi0.5 default flow steps, and a few-step target

# Peaks from vendor whitepapers and datasheets.
# GeForce cards run BF16 with FP32 accumulate at half rate, INT8 at full rate.
GPUS = {
    # RTX 3090 figures x 84/82 SMs x 1860/1695 MHz: https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf
    "RTX 3090 Ti": Gpu(bf16_flops_per_s=80e12, int8_ops_per_s=320e12, bandwidth_bytes_per_s=1008e9),
    # https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
    "RTX 4080": Gpu(bf16_flops_per_s=97.5e12, int8_ops_per_s=389.9e12, bandwidth_bytes_per_s=716.8e9),
    # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    "A100 80GB SXM": Gpu(bf16_flops_per_s=312e12, int8_ops_per_s=624e12, bandwidth_bytes_per_s=2039e9),
}

FORMATS = {
    "BF16": Format(weight_bits=16, activation_bits=16),
    "W8A8": Format(weight_bits=8, activation_bits=8),
    "W4A16": Format(weight_bits=4, activation_bits=16),
    "W4A8": Format(weight_bits=4, activation_bits=8),
}

# Shapes from openpi: SigLIP So400m/14, gemma_2b, gemma_300m.
# https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/siglip.py
# https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py
SIGLIP = Transformer(layers=27, width=1152, mlp_width=4304, heads=16, kv_heads=16, head_dim=72, gated_mlp=False)
SIGLIP_IMAGE_SIZE = 224
SIGLIP_PATCH_SIZE = 14
TOKENS_PER_IMAGE = (SIGLIP_IMAGE_SIZE // SIGLIP_PATCH_SIZE) ** 2
GEMMA_2B = Transformer(layers=18, width=2048, mlp_width=16384, heads=8, kv_heads=1, head_dim=256, gated_mlp=True)
PI05_ACTION_EXPERT = Transformer(
    layers=18, width=1024, mlp_width=4096, heads=8, kv_heads=1, head_dim=256, gated_mlp=True, adaptive_norm=True
)

# text_tokens is an estimated prompt length.
PI05_LIBERO = Workload(cameras=2, text_tokens=48, action_tokens=50)


def stage_costs(workload: Workload, action_expert_model: Transformer) -> Stages:
    image_tokens = workload.cameras * TOKENS_PER_IMAGE
    prefix_tokens = image_tokens + workload.text_tokens

    image_attention = SIGLIP.layer_attention_flops(TOKENS_PER_IMAGE, TOKENS_PER_IMAGE)
    vision_encoder = StageCost(
        linear_flops=2 * image_tokens * SIGLIP.token_params,
        attention_flops=workload.cameras * SIGLIP.layers * image_attention,
        weight_params=SIGLIP.weight_params,
        kv_cache_bytes=0,
    )

    # Nothing reads the last layer's output, only its K and V, so that layer stops after its K/V projection.
    full_layers = GEMMA_2B.layers - 1
    prefix_params = full_layers * GEMMA_2B.layer_params + GEMMA_2B.kv_params
    llm = StageCost(
        linear_flops=2 * prefix_tokens * prefix_params,
        attention_flops=full_layers * GEMMA_2B.layer_attention_flops(prefix_tokens, prefix_tokens),
        weight_params=prefix_params,
        kv_cache_bytes=0,
    )

    keys = prefix_tokens + workload.action_tokens
    action_attention = action_expert_model.layer_attention_flops(workload.action_tokens, keys)
    action_expert = StageCost(
        linear_flops=2 * workload.action_tokens * action_expert_model.token_params,
        attention_flops=action_expert_model.layers * action_attention,
        weight_params=action_expert_model.weight_params,
        kv_cache_bytes=GEMMA_2B.layers * GEMMA_2B.kv_width * prefix_tokens * KV_CACHE_BYTES_PER_VALUE,
    )

    return Stages(vision_encoder, llm, action_expert)


def latency_floor(cost: StageCost, gpu: Gpu, fmt: Format) -> LatencyFloor:
    linear_peak = gpu.int8_ops_per_s if fmt.activation_bits == 8 else gpu.bf16_flops_per_s
    compute = cost.linear_flops / linear_peak + cost.attention_flops / gpu.bf16_flops_per_s

    # Activations are too small at batch 1 to move the memory floor, so only weights and the KV cache count.
    memory = (cost.weight_params * fmt.weight_bits / 8 + cost.kv_cache_bytes) / gpu.bandwidth_bytes_per_s

    return LatencyFloor(ms=1e3 * max(compute, memory), compute_bound=compute > memory)


def main() -> None:
    stages = stage_costs(PI05_LIBERO, PI05_ACTION_EXPERT)
    console = Console()

    for gpu_name, gpu in GPUS.items():
        table = Table(title=f"{gpu_name} latency floor (ms), lower is better. C = compute-bound, M = memory-bound")
        table.add_column("Format")
        for column in ["Vision encoder", "LLM", "Action expert per step"]:
            table.add_column(column, justify="right")
        for steps in DENOISING_STEPS:
            table.add_column(f"Total, {steps} steps", justify="right")

        for format_name, fmt in FORMATS.items():
            vision_encoder = latency_floor(stages.vision_encoder, gpu, fmt)
            llm = latency_floor(stages.llm, gpu, fmt)
            action_expert = latency_floor(stages.action_expert, gpu, fmt)

            floors = (vision_encoder, llm, action_expert)
            cells = [f"{floor.ms:.2f} {'C' if floor.compute_bound else 'M'}" for floor in floors]
            totals = [f"{vision_encoder.ms + llm.ms + steps * action_expert.ms:.1f}" for steps in DENOISING_STEPS]
            table.add_row(format_name, *cells, *totals)

        console.print(table)


if __name__ == "__main__":
    main()
