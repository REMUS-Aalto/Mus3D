import torch

from query_conditioned_dit.trainer import LatentFlowTrainer
from launch import run

if __name__ == "__main__":
    # TF32 for the frozen encoders (PaSST, DAC-VAE), which run outside the
    # DiT's bf16 autocast. Scoped here so other models keep their numerics.
    torch.set_float32_matmul_precision("high")

    # Only the DiT is trained; the frozen encoders never see a gradient, and
    # the DAC-VAE's weight_norm is not deepcopyable (EMA would fail on it).
    run(
        "DiT_separator.yaml",
        LatentFlowTrainer,
        params_fn=lambda m: m.backbone.parameters(),
        report_fn=lambda m: m.backbone,
        ema_fn=lambda m: m.backbone,
    )
    print("[Training finished.]")
