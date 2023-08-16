
from transformers import TrainingArguments
from evaluation.run_mlm import ModelArguments, DataTrainingArguments, main

model_args = ModelArguments(
    model_name_or_path="roberta-base",
    cache_dir="../huggingface_cache",
)
data_args = DataTrainingArguments(
    dataset_name="the_pile",
    # remove train_file argument for baseline training
    # train_file="../data/CMS100_CS100_TL0.1_TH0.7/filtered_data.json",
    validation_file="../data/the_pile_valid_1000.json",
    streaming=True,
    max_seq_length=512,
    max_eval_samples=1000,
)
train_args = TrainingArguments(
    run_name="CMS100_CS100_TL0.1_TH0.7",
    output_dir="../dump/pile_baseline",
    logging_dir="../wandb",
    num_train_epochs=1,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=500,
    max_steps=10000000,
    logging_steps=25,
    seed=42,
    # overwrite_output_dir=True,
    # resume_from_checkpoint=True,
    fp16=True,
)

main(model_args, data_args, train_args)