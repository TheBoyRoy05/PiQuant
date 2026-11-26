"""Run lerobot-eval while recording the token counts pi0.5 actually sees.

The roofline needs the real image, prompt and action token counts. LeRobot pads the prompt to
`tokenizer_max_length`, so this records both the padded length the model computes over and the
number of real (unpadded) tokens. Usage: same arguments as lerobot-eval.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.scripts import lerobot_eval

OUTPUT_PATH = Path("outputs/token_counts.json")

logger = logging.getLogger(__name__)


class PrefixShape(NamedTuple):
    image_tokens: int
    prompt_tokens_padded: int
    prompt_tokens_real: int


def main() -> None:
    prefix_shapes: Counter[PrefixShape] = Counter()
    suffix_lengths: Counter[int] = Counter()

    _patch_embed_prefix(prefix_shapes)
    _patch_embed_suffix(suffix_lengths)

    try:
        lerobot_eval.main()
    except (Exception, KeyboardInterrupt):
        logger.warning("eval aborted; token counts in %s are partial", OUTPUT_PATH)
        raise
    finally:
        _write_counts(prefix_shapes, suffix_lengths)


def _patch_embed_prefix(prefix_shapes: Counter[PrefixShape]) -> None:
    original_embed_prefix = PI05Pytorch.embed_prefix

    def recording_embed_prefix(self, images, img_masks, tokens, masks):
        prefix = original_embed_prefix(self, images, img_masks, tokens, masks)

        embeddings, *_ = prefix
        prompt_tokens_padded = tokens.shape[1]
        image_tokens = embeddings.shape[1] - prompt_tokens_padded
        for prompt_tokens_real in masks.sum(dim=1).tolist():
            shape = PrefixShape(image_tokens, prompt_tokens_padded, prompt_tokens_real)
            prefix_shapes[shape] += 1

        return prefix

    PI05Pytorch.embed_prefix = recording_embed_prefix


def _patch_embed_suffix(suffix_lengths: Counter[int]) -> None:
    original_embed_suffix = PI05Pytorch.embed_suffix

    def recording_embed_suffix(self, noisy_actions, timestep):
        suffix = original_embed_suffix(self, noisy_actions, timestep)

        batch_size, action_tokens = noisy_actions.shape[:2]
        # The suffix is embedded once per denoising step, so this counts samples x steps.
        suffix_lengths[action_tokens] += batch_size

        return suffix

    PI05Pytorch.embed_suffix = recording_embed_suffix


def _write_counts(prefix_shapes: Counter[PrefixShape], suffix_lengths: Counter[int]) -> None:
    counts = {
        "prefix_shapes": [
            {**shape._asdict(), "samples": samples}
            for shape, samples in sorted(prefix_shapes.items())
        ],
        "suffix_lengths": [
            {"action_tokens": length, "sample_steps": sample_steps}
            for length, sample_steps in sorted(suffix_lengths.items())
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(counts, indent=2))
    logger.info(
        "token counts written to %s (%d prefix shapes, %d suffix lengths)",
        OUTPUT_PATH,
        len(prefix_shapes),
        len(suffix_lengths),
    )


if __name__ == "__main__":
    main()
