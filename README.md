# PiQuant

Where does Pi0.5 spend its inference time on consumer GPUs, and does quantizing the bottleneck turn into a real speedup? DSC 180A group D38, mentored by [Zhijian Liu](https://github.com/zhijian-liu) ([Z-Lab](https://github.com/z-lab)).

## Data

- Model: [`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned), Pi0.5 fine-tuned on LIBERO (about 14.5 GB, downloaded on first run)
- Benchmark: [LIBERO](https://libero-project.github.io/) simulated manipulation suites, run through [LeRobot](https://huggingface.co/docs/lerobot/libero)

## Requirements

- Linux with an NVIDIA GPU, since LIBERO requires Linux
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone git@github.com:TheBoyRoy05/PiQuant.git
cd PiQuant
uv sync
```

## Run

Evaluate Pi0.5 on one LIBERO suite (default `libero_spatial`, 10 episodes per task):

```bash
scripts/eval_libero.sh libero_spatial 10
```
