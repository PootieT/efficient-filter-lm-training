# CS543-final-project
some dataset selection algorithm. 

For details please see our report `Filtering_for_LMs.pdf`

# Data Selection

Checkout `filtering` directory.

# Evaluation

## Quick Evaluation (KL-Reduction)

```bash
python evaluation/run_quick_eval.py \
    --sampled_data_path /path/to/data \
    --seed 42 \
```

## Final Evaluation (Finetune -> GLUE)
For the final evaluation, we evaluate whether our dataset selection
method can improve performance by having the model continued to be
pretrained on our model. We then use the resulting model to finetune 
for a suite of standard NLP tasks for average performance. 

To continue pretraining a model with MLM objective on given dataset:
```bash
python evaluation/run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /dump/test-mlm
```

To finetune and evaluate a model on a single task in GLUE benchmark:
```bash
export TASK_NAME=mrpc #cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli

python evaluation/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```
According to [huggingface documentation](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)
All tasks a ran with above commands (with an exception for MRPC and WNLI which are tiny and where we used 5 epochs instead of 3)

To make things easier, you can evaluate the whole benchmark with our compiled script:
```bash
./evaluate/run_glue_all.sh
```