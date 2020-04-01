#model_path=/data/zhanghaipeng/base_bert_model/bert-base-chinese
model_path=/nfs/users/zhanghaipeng/data/pt_bert_models/roberta-base
#model_path=/nfs/users/zhanghaipeng/data/pt_bert_models/bert-base-chinese
data_dir=/nfs/users/zhanghaipeng/general_ner/mrc_data/
#data_dir=/data/zhanghaipeng/ner_data/
expr=7
cuda=1
CUDA_VISIBLE_DEVICES=$cuda python run_ner.py \
	--data_dir $data_dir/chinese/merge_boundary_ontonotes \
	--model_type bert \
	--labels $data_dir/labels/aigen_labels.txt \
	--bd_labels $data_dir/labels/aigen_boundary_labels.txt \
	--model_name_or_path $model_path \
	--output_dir $data_dir/models/$expr \
	--log_dir $data_dir/logs/$expr \
	--num_train_epochs 10 \
	--per_gpu_train_batch_size 64 \
	--per_gpu_eval_batch_size 64 \
	--max_seq_length  80 \
	--seed 100 \
	--learning_rate 3e-5 \
	--do_train \
	--evaluate_during_training \
	--overwrite_output_dir \
	--logging_steps 50 \
	--save_steps 50 \
