"""Time each pi0.5 stage with CUDA events, and summarize the timings across calls."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import torch
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch

from piquant.roofline import TOKENS_PER_IMAGE


class Stage(StrEnum):
    """The timed parts of a call, in report order."""

    VISION = "vision encoder"
    LLM = "llm prefix"
    DENOISING = "denoising"
    EXPERT = "action expert"  # the expert's forward pass alone, inside each denoising step
    STEP = "step"  # one denoising step
    OTHER = "other"  # what the total spends outside the timed stages
    TOTAL = "total"


CallTimes = dict[str, list[float]]  # each timed call's ms, by stage


@dataclass(frozen=True)
class Prefix:
    image_slots: int
    blank_image_slots: int  # empty cameras: masked out of attention but still encoded and run through the LLM
    tokens_per_image: int
    text_tokens: int  # padded to tokenizer_max_length, so all of them are computed

    @property
    def tokens(self) -> int:
        return self.image_slots * self.tokens_per_image + self.text_tokens

    @classmethod
    def from_embeddings(
        cls, images: list[torch.Tensor], img_masks: list[torch.Tensor], tokens: torch.Tensor, prefix_embs: torch.Tensor
    ) -> Prefix:
        """The prefix a call actually built. Every image slot embeds to the same number of tokens."""
        tokens_per_image = (prefix_embs.shape[1] - tokens.shape[1]) // len(images)
        blank = sum(1 for mask in img_masks if not bool(mask.any()))
        return cls(len(images), blank, tokens_per_image, tokens.shape[1])

    @classmethod
    def from_config(cls, config: PI05Config) -> Prefix:
        """The prefix as the policy builds it, for compiled runs, whose stages cannot be hooked."""
        return cls(len(config.image_features), config.empty_cameras, TOKENS_PER_IMAGE, config.tokenizer_max_length)


@dataclass(frozen=True)
class Summary:
    median_ms: float
    p90_ms: float


def measure(
    policy: PI05Policy, batch: dict[str, torch.Tensor], warmup: int, iters: int, breakdown: bool
) -> tuple[list[CallTimes], Prefix | None]:
    """Time iters calls after warmup ones. breakdown also times each stage and reads the prefix the calls build."""
    timer = _StageTimer()
    built_prefix = _time_stages(policy.model, timer) if breakdown else lambda: None
    timer.wrap(policy, "predict_action_chunk", _always(Stage.TOTAL))

    for _ in range(warmup):
        policy.predict_action_chunk(batch)
    timer.read()

    calls = []
    for _ in range(iters):
        policy.predict_action_chunk(batch)
        calls.append(timer.read())
    return calls, built_prefix()


def _time_stages(model: PI05Pytorch, timer: _StageTimer) -> Callable[[], Prefix | None]:
    """Time the vision encoder, LLM prefix, denoising steps and expert passes. Returns a reader for the prefix
    the last call built."""
    prefix_inputs = None
    embed_prefix = model.embed_prefix

    def capture(images, img_masks, tokens, masks):
        nonlocal prefix_inputs
        out = embed_prefix(images, img_masks, tokens, masks)
        # Keep the tensors, not their values: reading a mask here would sync inside the timed vision encoder.
        prefix_inputs = (images, img_masks, tokens, out[0])
        return out

    def forward_stage(*args, inputs_embeds, **kwargs) -> Stage | None:
        # The prefix pass runs without suffix embeddings, and each step's expert pass without prefix ones.
        prefix_embs, suffix_embs = inputs_embeds
        if suffix_embs is None:
            return Stage.LLM
        if prefix_embs is None:
            return Stage.EXPERT
        return None

    model.embed_prefix = capture
    # embed_prefix runs SigLIP on every image slot; the text embedding lookup it also does is negligible.
    timer.wrap(model, "embed_prefix", _always(Stage.VISION))
    timer.wrap(model.paligemma_with_expert, "forward", forward_stage)
    timer.wrap(model, "denoise_step", _always(Stage.DENOISING))
    return lambda: Prefix.from_embeddings(*prefix_inputs) if prefix_inputs else None


def _always(stage: Stage) -> Callable[..., Stage]:
    return lambda *args, **kwargs: stage


class _StageTimer:
    """Brackets wrapped methods with CUDA events. Events are read after each call, so timing adds no syncs."""

    def __init__(self):
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def wrap(self, obj: object, method: str, stage_of: Callable[..., str | None]) -> None:
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

    def read(self) -> CallTimes:
        """Each timed call's ms, per stage, since the last read."""
        torch.cuda.synchronize()
        ms = {stage: [start.elapsed_time(end) for start, end in pairs] for stage, pairs in self._events.items()}
        self._events.clear()
        return ms


def summarize(calls: list[CallTimes]) -> dict[str, Summary]:
    """Median and p90 across calls of each stage's time per call, summed over e.g. its denoising steps.

    A broken-down run also gets Stage.OTHER and Stage.STEP.
    """
    per_call = [{stage: sum(ms) for stage, ms in call.items()} for call in calls]
    breakdown = Stage.DENOISING in per_call[0]
    if breakdown:
        for call in per_call:
            call[Stage.OTHER] = call[Stage.TOTAL] - call[Stage.VISION] - call[Stage.LLM] - call[Stage.DENOISING]

    summary = {stage: _describe([call[stage] for call in per_call]) for stage in per_call[0]}
    if breakdown:
        # From single steps, not their sum per call: the sum carries the occasional stall, which eager
        # launch-bound runs show as steps several times slower than the rest.
        summary[Stage.STEP] = _describe([ms for call in calls for ms in call[Stage.DENOISING]])
    return summary


def _describe(values: list[float]) -> Summary:
    # inclusive keeps p90 within the samples; the default exclusive method extrapolates past the max on few calls.
    p90 = statistics.quantiles(values, n=10, method="inclusive")[-1] if len(values) > 1 else values[0]
    return Summary(statistics.median(values), p90)
