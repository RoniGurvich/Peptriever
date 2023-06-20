from peptriever.finetuning.config import FinetuningConfig
from peptriever.finetuning.evaluate_peptriever import evaluate_binding
from peptriever.finetuning.finetuning import TrainParams, finetune_protein_binding_model
from peptriever.reproducibility import set_seed


def run_finetuning():
    config = FinetuningConfig()
    set_seed(seed=999)

    model_name = finetune_protein_binding_model(
        config=config,
        train_params=TrainParams(
            pretrained_weights=(
                "base_protein_30_2023-06-17T19:30:56.000452",
                "base_protein_300_2023-06-17T19:30:56.000452",
            ),
            epochs=80,
            warmup=10,
            cooldown=50,
        ),
    )
    model_name = finetune_protein_binding_model(
        config=config,
        train_params=TrainParams(
            resume_training=model_name,
            warmup=1,
            cooldown=48,
            epochs=50,
        ),
    )

    evaluate_binding(
        config=config,
        model_name=model_name,
    )


if __name__ == "__main__":
    run_finetuning()
