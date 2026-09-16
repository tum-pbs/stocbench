---
eyebrow: Preprint 2026
title: StocBench: A Benchmark for Generative Modeling of Stochastic Dynamics
authors: Sebastian Pfister, Benjamin Holzschuh, Nils Thuerey
affiliation: Technical University of Munich
links: [arXiv](https://arxiv.org/abs/2608.22309) [Dataset](https://huggingface.co/datasets/pfistse/stocbench-data) [BibTeX](https://arxiv.org/bibtex/2608.22309) **[GitHub](https://github.com/tum-pbs/stocbench)** **[Pip package](https://pypi.org/)**
---
> We benchmark transport-based generative models and few-step distillation methods for probabilistic forecasting of stochastic fluid flows, with a focus on limited inference budgets. All methods are evaluated on a two-dimensional Kolmogorov flow with stochastic forcing: we measure the one-step conditional distribution against large simulated reference ensembles and check whether the invariant measure is preserved during autoregressive rollouts via the enstrophy spectrum. A deterministic control task with observed forcing separates aleatoric from epistemic uncertainty.

## The benchmark

Following Chen et al. (2024), we simulate incompressible Navier–Stokes equations on the two-dimensional torus with stochastic forcing. Every method learns the next-state distribution $p(\omega_{t+\Delta t} \mid \omega_t)$ and is evaluated on two variants:
a. **Stochastic.** The forcing is unobserved; the uncertainty in the next state is irreducible and the model must capture a genuine conditional distribution.
b. **Deterministic control.** The forcing is part of the input, so the next state is fully determined and models should collapse to a single prediction.
|[1fr] ![Left: one-step prediction ensemble for a given initial condition. Right: pointwise std of the ensemble.](assets/samples.mp4 poster=assets/samples_poster.png, assets/std.png)

## Metrics

|[1fr] ### One-step distribution
|[3fr] - **Mean error** Relative L2 error of the predicted conditional mean.
- **Std error** Relative L2 error of the predicted conditional standard deviation.
- **Energy distance** Distributional mismatch beyond the first two moments.

|[1fr] ### Invariant measure
|[3fr] - **Enstrophy spectrum error** Deviation of the ensemble-averaged enstrophy spectrum during a 50-step autoregressive rollout.

## Baselines

![Dotted: 1 NFE, dashed: 2 NFEs, solid: evaluated at every budget on the x-axis.](chart:baselines)

## Stochastic task

![**One-step distribution.** Relative L2 error of the predicted conditional mean (left) and std (right) vs. inference budget.](chart:mean, chart:std)

| 1. **Flow matching is most accurate at high budgets.** It has the lowest mean error at every budget and the lowest std error and energy distance once the budget is large.
| 2. **DPM-2 is the strongest multi-step method at low budgets.** Its second-order exponential integrator gives the best std error of all multi-step methods at about 20 NFEs, but its accuracy plateaus early.
| 3. **Distillation is competitive with one or two NFEs.** ADD-FM matches the std error of flow matching with a single evaluation. Consistency distillation has a clearly higher mean error, but its std error is on par with the diffusion samplers at 400 NFEs.

![**Invariant measure.** Error in the average enstrophy spectrum over a 50-step autoregressive rollout vs. inference budget (left), and enstrophy ratio per wavenumber after 50 rollout steps, where 1.0 preserves the ground-truth spectrum (right).](chart:enstrophy, chart:ratio)

| 4. **Deterministic sampling damps the spectrum.** DDIM, DPM-2, and DDPM share one network, yet their rollout enstrophy errors differ by roughly a factor of four: the deterministic samplers converge to a damped invariant measure, while DDPM's stochastic updates preserve small-scale enstrophy.
| 5. **Flow matching, ADD-FM, and CD-2 preserve the spectrum best.** FM has the lowest enstrophy error among the multi-step methods at high budgets, and the two distilled models reach the same level with one or two NFEs.

## Deterministic control task

![**One-step distribution.** Mean error, i.e. the deviation from the ground-truth next-step state (left), and residual std error, where the target std is zero (right).](chart:det_mean, chart:det_std)

6. **Performance does not translate between settings.** With the forcing observed, the multi-step methods form a tight cluster with low error, while the distilled models are least accurate and retain residual variability that is epistemic rather than physical.

![**Invariant measure.** Error in the average enstrophy spectrum along a 50-step rollout (left) and enstrophy ratio per wavenumber after 50 steps, where 1.0 preserves the ground-truth spectrum (right).](chart:det_rollout, chart:det_ratio)

7. **The diffusion sampler ranking reverses.** The stochastic updates that let DDPM preserve the spectrum on the stochastic task now add excess enstrophy, and DDIM and DPM-2 preserve it better.

## Benchmark your model in a few lines

Datasets for both Kolmogorov-flow variants, precomputed ground-truth ensembles, all four evaluation metrics, and reference implementations of all eight baselines ship in one Python package.

[Code](https://github.com/tum-pbs/stocbench) [Pip package](https://pypi.org/)