data_dir=/nfs/users/zhanghaipeng/data/toutiao_news/B
output_dir=/nfs/users/zhanghaipeng/data/train/news_cls
model_path=/nfs/users/zhanghaipeng/data/pt_bert_models/bert-base-chinese/
pre_expr=0
ckpt=49000
cur_expr=2
tb_path=$output_dir/$cur_expr/runs
cuda=2
CUDA_VISIBLE_DEVICES=$cuda python run_glue.py \
	--model_type bert \
	--model_name_or_path $output_dir/$pre_expr/ \
	--tb_path $tb_path \
	--task_name news_cls \
	--do_train \
	--do_eval \
	--data_dir $data_dir \
	--max_seq_length 30 \
	--per_gpu_eval_batch_size 100   \
	--per_gpu_train_batch_size 100  \
	--overwrite_output_dir \
	--learning_rate 2e-5 \
	--num_train_epochs 20.0 \
	--output_dir $output_dir/$cur_expr/ \
	--save_steps 100 \
	--logging_steps 100 \
