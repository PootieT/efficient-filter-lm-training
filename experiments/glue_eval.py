
from transformers import TrainingArguments
from evaluation.run_glue import ModelArguments, DataTrainingArguments, main

model_args = ModelArguments(
    model_name_or_path="../dump/pile_finetune_debug",
    cache_dir="../huggingface_cache",
)
data_args = DataTrainingArguments(
    dataset_name="the_pile",
    max_seq_length=128,
    task_name="cola"
)
train_args = TrainingArguments(
    run_name="pile_finetune_debug-cola",
    output_dir="../dump/pile_finetune_debug/cola",
    logging_dir="../wandb",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    do_train=True,
    do_eval=True,
    overwrite_output_dir=True,
)

main(model_args, data_args, train_args)