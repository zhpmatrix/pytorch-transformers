model_path=/data/share/zhanghaipeng/data/pt_bert_models/bert-base-chinese
data_dir=/nfs/users/zhanghaipeng/general_ner/data
expr=1
cuda=3

CUDA_VISIBLE_DEVICES=$cuda python run_ner.py \
	--data_dir $data_dir/chinese \
	--model_type bert \
	--labels $data_dir/labels/aigen_labels.txt \
	--model_name_or_path $model_path \
	--output_dir $data_dir/logs/$expr \
	--num_train_epochs 3 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--max_seq_length  40 \
	--seed 100 \
	--learning_rate 3e-5 \
	--do_train \
	--evaluate_during_training \
	--overwrite_output_dir \
	--logging_steps 10 \
	--save_steps 20 \
