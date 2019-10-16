data_dir=/data/share/zhanghaipeng/data/yelp/style_transfer/
cuda=0
expr=0
ckpt=1200
CUDA_VISIBLE_DEVICES=$cuda python run_glue.py \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--task_name stmi \
	--do_eval \
	--do_lower_case \
	--data_dir $data_dir \
	--max_seq_length 128 \
	--evaluate_during_training \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir train/$expr/checkpoint-$ckpt/ \
	--save_steps 200 \
	--logging_steps 100 
