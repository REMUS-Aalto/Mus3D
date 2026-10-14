from source_embedding_generator.trainer import RecursiveFMTrainer
from launch import run

if __name__ == "__main__":
    # Frozen encoders throughout: only the DiT trains.
    run(
        "SEG.yaml",
        RecursiveFMTrainer,
        params_fn=lambda m: m.backbone.parameters(),
        report_fn=lambda m: m.backbone,
        ema_fn=lambda m: m.backbone,
    )
    print("[Training finished.]")
