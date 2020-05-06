model_path=/nfs/users/zhanghaipeng/data/pt_bert_models/roberta-base
data_dir=/nfs/users/zhanghaipeng/data/kuaishou/simi/data_with_label_balanced
task_name=simi_kuaishou
expr=0
cuda=1
ckpt=$1
CUDA_VISIBLE_DEVICES=$cuda python run_glue.py \
	--data_dir $data_dir \
	--task_name $task_name \
	--model_type bert \
	--model_name_or_path $model_path \
	--output_dir $data_dir/models/$expr/checkpoint-$ckpt \
	--per_gpu_eval_batch_size 64 \
	--max_seq_length  60 \
	--seed 100 \
	--do_eval \
	--overwrite_output_dir \
