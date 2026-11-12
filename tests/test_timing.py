import pytest
import torch

from piquant.timing import Prefix, _StageTimer, _describe, summarize


def test_prefix_counts_blank_cameras_and_padded_text():
    images = [torch.zeros(1, 3, 224, 224)] * 3
    img_masks = [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)]
    tokens = torch.zeros(1, 200, dtype=torch.long)
    prefix_embs = torch.zeros(1, 3 * 256 + 200, 8)

    prefix = Prefix.from_embeddings(images, img_masks, tokens, prefix_embs)

    assert prefix == Prefix(image_slots=3, blank_image_slots=1, tokens_per_image=256, text_tokens=200)
    assert prefix.tokens == 968


def test_summarize_sums_steps_and_puts_untimed_time_in_other():
    def call(vision, llm, steps, total):
        return {"vision encoder": [vision], "llm prefix": [llm], "denoising": steps, "total": [total]}

    calls = [call(5.0, 30.0, [10.0, 10.0], 60.0), call(7.0, 32.0, [11.0, 11.0], 66.0), call(6.0, 31.0, [9.0, 12.0], 63.0)]

    summary = summarize(calls)

    assert summary["denoising"].median_ms == pytest.approx(21.0)
    assert summary["other"].median_ms == pytest.approx(5.0)
    assert summary["total"].median_ms == pytest.approx(63.0)
    assert summary["step"].median_ms == pytest.approx(10.5)  # from single steps, not their sum per call
    assert "other" not in calls[0]  # the input is not modified


def test_summarize_total_only():
    summary = summarize([{"total": [40.0]}, {"total": [42.0]}])

    assert set(summary) == {"total"}
    assert summary["total"].median_ms == pytest.approx(41.0)


def test_p90_stays_within_the_samples():
    summary = _describe([float(ms) for ms in range(1, 11)])

    assert summary.p90_ms == pytest.approx(9.1)
    assert _describe([4.0]).p90_ms == 4.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_stage_timer_times_only_the_selected_calls():
    class Worker:
        def run(self, n):
            return torch.ones(n, n, device="cuda") @ torch.ones(n, n, device="cuda")

    worker = Worker()
    timer = _StageTimer()
    timer.wrap(worker, "run", lambda n: "big" if n > 1 else None)

    worker.run(1)
    worker.run(512)
    worker.run(512)
    ms = timer.read()

    assert set(ms) == {"big"}
    assert len(ms["big"]) == 2
    assert all(t > 0 for t in ms["big"])
    assert timer.read() == {}
