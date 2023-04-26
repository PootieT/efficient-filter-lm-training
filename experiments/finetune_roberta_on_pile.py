
from transformers import TrainingArguments
from evaluation.run_mlm import ModelArguments, DataTrainingArguments, main

model_args = ModelArguments(
    model_name_or_path="roberta-base",
    cache_dir="../huggingface_cache",
)
data_args = DataTrainingArguments(
    dataset_name="the_pile",
    # remove train_file argument for baseline training
    train_file="../dump/CMS100_FS10000_TL0.01_TH0.99_PH0.1/filtered_text.json",
    streaming=True,
    max_seq_length=512,
    max_eval_samples=1000,
)
train_args = TrainingArguments(
    run_name="baseline",
    output_dir="../dump/pile_baseline",
    logging_dir="../wandb",
    num_train_epochs=1,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=500,
    max_steps=10000000,
    logging_steps=25,
    seed=42,
    overwrite_output_dir=True,
    fp16=True,
)

main(model_args, data_args, train_args)