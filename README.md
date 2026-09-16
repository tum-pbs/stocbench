# StocBench: A Benchmark for Generative Modeling of Stochastic Dynamics

<p align="center">
  <a href="https://arxiv.org/abs/2608.22309"><img src="https://img.shields.io/badge/arXiv-2608.22309-b31b1b" alt="arXiv"></a>
  <a href="https://pfistse.github.io/stocbench-page/"><img src="https://img.shields.io/badge/project-page-1f6feb" alt="Project page"></a>
  <a href="https://huggingface.co/datasets/pfistse/stocbench-data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-stocbench--data-ffd21e" alt="Dataset"></a>
  <a href="https://huggingface.co/pfistse/stocbench-ckpts"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20checkpoints-stocbench--ckpts-ffd21e" alt="Checkpoints"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

**Authors:** Sebastian Pfister, Benjamin Holzschuh, Nils Thuerey

StocBench benchmarks generative models for probabilistic forecasting of a stochastically forced 2D Kolmogorov flow. Every model learns the next-state distribution p(ω<sub>t+Δt</sub> | ω<sub>t</sub>) and is evaluated on two tasks:

- **Stochastic** (`experiment=stoc`): the forcing is unobserved. The next state is genuinely uncertain, and the model must capture the conditional distribution.
- **Deterministic control** (`experiment=det`): the forcing is part of the input. The next state is fully determined, and the model should collapse to a single prediction.

The one-step distribution is scored against large simulated reference ensembles (mean error, std error, energy distance). Autoregressive rollouts are checked for preserving the invariant measure via the enstrophy spectrum. Baselines: diffusion with DDIM, DDPM and DPM-Solver-2 samplers, flow matching, stochastic interpolants, consistency distillation and adversarial distillation. Datasets and checkpoints are downloaded from Hugging Face on first use.

## Install

```bash
pip install -e .
```

Python 3.11 or newer. `pip install -e ".[solver]"` adds the JAX solver that generates the datasets.

## Usage

`stocbench eval`, `stocbench train` and `stocbench plot` take Hydra overrides.

Evaluate a published checkpoint under the paper protocol:

```bash
stocbench eval experiment=det model=fm hydra.run.dir=results/det/fm
```

Train a model and plot finished runs:

```bash
stocbench train experiment=det model=fm trainer.max_epochs=800
stocbench plot results/det/fm results/det/si --plots-config src/stocbench/configs/plotting/plots/stocbench/det.yaml --output-dir plots/det
```

| Config group | Options |
|---|---|
| `experiment` | `det`, `stoc` |
| `model` | `dm`, `fm`, `si`, `edm`, `cd`, `add_fm` |

- `ckpt_path` defaults to the published checkpoint of the selected experiment and model.
- The paper protocol (`evaluate=stocbench`, the default) sweeps 13 budgets with 10-step rollouts, then repeats the rollout metrics over 50 steps at the last budget (`r50_*` artifacts).
- `cd` and `add_fm` distill `edm` and `fm`, set by `model.teacher_ckpt`. Multi-GPU `add_fm` training needs `trainer.strategy=ddp_find_unused_parameters_true`.
- Datasets and checkpoints are cached in `~/.cache/stocbench`. `STOCBENCH_DATA_DIR` and `STOCBENCH_CKPT_DIR` move the caches.
- An evaluation writes `config.yaml`, `summary.md`, `artifacts/` and `plots/` into its run directory. `stocbench.BenchmarkResult(run_dir)` reads them back.

## Reproducing the paper

[scripts/reproduce_stocbench.sh](scripts/reproduce_stocbench.sh) lists every evaluation: eight sampler configurations per experiment, then the figures. Submit its lines to a cluster rather than running it in one shot.

## Datasets

The solver in [solver/](solver) generated the data:

```bash
python -m solver.create_incns_stoc_dataset 42 43 44 --out data/incns_stoc --warmup-frames 100 --traj-frames 200 --batch-size 500 --num-test-seeds 48 --stats-samples 5000 --out-grid 64
python -m solver.create_incns_det_dataset 42 43 44 45 --out data/incns_det --warmup-frames 100 --traj-frames 200 --batch-size 500 --out-grid 64
python -m solver.enstrophy_spec data/incns_stoc
```

Use a local copy with `data.root=data/incns_stoc`.

## Citation

```bibtex
@misc{pfister2026stocbenchbenchmarkgenerativemodeling,
      title={StocBench: A Benchmark for Generative Modeling of Stochastic Dynamics}, 
      author={Sebastian Pfister and Benjamin Holzschuh and Nils Thuerey},
      year={2026},
      eprint={2608.22309},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.22309}, 
}
```

## License

MIT, see [LICENSE](LICENSE).

