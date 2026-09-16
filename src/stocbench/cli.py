"""Thin dispatcher to the hydra entry points and the plot driver."""

import sys

USAGE = """usage: stocbench COMMAND [args...]

commands (train/eval take hydra overrides):
  train   e.g. stocbench train experiment=det model=fm trainer.max_epochs=100
  eval    e.g. stocbench eval experiment=stoc model=dm evaluate=smoke
  plot    e.g. stocbench plot results/stoc/si results/stoc/fm --output-dir plots  (see stocbench plot --help)
"""


def app() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("train", "eval", "plot"):
        sys.exit(USAGE)
    sys.argv[:2] = [f"{sys.argv[0]} {cmd}"]
    if cmd == "train":
        from .train import main
    elif cmd == "eval":
        from .eval import main
    else:
        from .utils.plotting import main
    main()


if __name__ == "__main__":
    app()
