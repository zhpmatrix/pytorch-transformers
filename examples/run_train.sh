data_dir=data
train_file=$data_dir/train.txt
test_file=$data_dir/test.txt
model_path=base_model/roberta-wwm-ext-chinese/
cuda=3
expr=1

CUDA_VISIBLE_DEVICES=$cuda python run_lm_finetuning.py \
	--output_dir=train/$expr \
	--model_type=bert \
	--model_name_or_path=$model_path \
	--do_train \
	--train_data_file=$train_file \
	--do_eval \
	--eval_data_file=$test_file \
	--mlm \
	--block_size 128 \
	--per_gpu_train_batch_size 2 \
	--per_gpu_eval_batch_size 2 \
	--save_steps 50 \
	--logging_steps 50
