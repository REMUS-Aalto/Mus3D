import sys

from query_conditioned_bslocoformer.trainer import Trainer
from launch import run

if __name__ == "__main__":
    # `--config BSLocoformer_static` trains the static VDBGPO baseline.
    config = "BSLocoformer_separator.yaml"
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config = sys.argv[2] + ".yaml"
        del sys.argv[1:3]
    run(config, Trainer)
    print("[Training finished.]")
