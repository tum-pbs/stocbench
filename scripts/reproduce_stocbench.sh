#!/usr/bin/env bash
# The StocBench paper evaluations from the published checkpoints: eight sampler configurations per
# experiment under evaluate=stocbench, then the figures. Every evaluation takes hours on 16 A100s,
# so submit the lines to a cluster instead of running the script in one shot.
set -euo pipefail
cd "$(dirname "$0")/.."  # results/ and the plot configs are addressed relative to the repo root

EXP_SAMPLER=stocbench.models.diffusion.samplers.ExpSampler
DDIM_SAMPLER=stocbench.models.diffusion.samplers.DDIMSampler
# DPM-Solver-2 spends two network evaluations per step, so its budgets start at u:2.
DPM2_SCHEDULES='[u:2,u:3,u:4,u:5,u:6,u:7,u:8,u:9,u:10,u:20,u:30,u:40,u:50,u:60,u:70,u:80,u:90,u:100,u:200,u:300,u:400]'

bench() {  # bench EXPERIMENT RUN [overrides...]: evaluate=stocbench into results/EXPERIMENT/RUN
  local e=$1 run=$2
  shift 2
  stocbench eval "experiment=$e" evaluate=stocbench "hydra.run.dir=results/$e/$run" "$@"
}

for e in det stoc; do
  bench "$e" si       model=si
  bench "$e" fm       model=fm
  bench "$e" dpm2     model=dm model.name=dpm2 "model.sampler._target_=$EXP_SAMPLER" "evaluate.sampling_schedules=$DPM2_SCHEDULES"
  bench "$e" ddim_e00 model=dm model.name=ddim_e00 "model.sampler._target_=$DDIM_SAMPLER" +model.sampler.eta=0.0
  bench "$e" ddim_e10 model=dm model.name=ddim_e10 "model.sampler._target_=$DDIM_SAMPLER" +model.sampler.eta=1.0
  bench "$e" cd1      model=cd model.name=cd1 'evaluate.sampling_schedules=["at:[0]"]'
  bench "$e" cd2      model=cd model.name=cd2 'evaluate.sampling_schedules=["at:[0,2]"]'
  bench "$e" addfm    model=add_fm model.name=addfm 'evaluate.sampling_schedules=[u:1]'

  stocbench plot results/"$e"/{si,dpm2,ddim_e00,ddim_e10,fm,cd1,cd2,addfm} \
    --plots-config "src/stocbench/configs/plotting/plots/stocbench/$e.yaml" --output-dir "plots/$e"
done
