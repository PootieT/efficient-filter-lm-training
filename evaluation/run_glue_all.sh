#!/usr/bin/env bash

for TASK_NAME in cola sst2 mrpc stsb qqp mnli qnli rte wnli
do
  if [[ $TASK_NAME == "mrpc" ]] || [[ $TASK_NAME == "wnli" ]]; then
    EPOCHS=5
  else
    EPOCHS=3
  fi
  echo "============= running glue task: ${TASK_NAME} w/ ${EPOCHS} epochs ============="
  python run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs $EPOCHS \
    --output_dir /dump/$TASK_NAME/
done