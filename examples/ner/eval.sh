model_path=/data/share/zhanghaipeng/data/pt_bert_models/bert-base-chinese
data_dir=/nfs/users/zhanghaipeng/general_ner/data
expr=1
cuda=1
ckpt=200
CUDA_VISIBLE_DEVICES=$cuda python run_ner.py \
	--data_dir $data_dir/chinese \
	--model_type bert \
	--labels $data_dir/labels/aigen_labels.txt \
	--model_name_or_path $model_path \
	--output_dir $data_dir/logs/$expr/checkpoint-$ckpt \
	--per_gpu_eval_batch_size 32 \
	--max_seq_length  40 \
	--seed 100 \
	--learning_rate 3e-5 \
	--do_eval \
	--overwrite_output_dir \
	--logging_steps 100 \
	--save_steps 200 \
