model_path=/nfs/users/zhanghaipeng/data/pt_bert_models/roberta-base
data_dir=/nfs/users/zhanghaipeng/data/kuaishou/bucket
label_name=label.csv
task_name=kuaishou
expr=0
cuda=0
CUDA_VISIBLE_DEVICES=$cuda python run_glue.py \
	--data_dir $data_dir \
	--task_name $task_name \
	--model_type bert \
	--model_name_or_path $model_path \
	--output_dir $data_dir/models/$expr \
	--label_path $data_dir/label.csv \
	--num_train_epochs 400 \
	--per_gpu_train_batch_size 64 \
	--per_gpu_eval_batch_size 64 \
	--max_seq_length  60 \
	--seed 100 \
	--learning_rate 3e-5 \
	--do_train \
	--evaluate_during_training \
	--overwrite_output_dir \
	--logging_steps 50 \
	--save_steps 100 \
