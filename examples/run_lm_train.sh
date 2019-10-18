data_dir=/data/share/zhanghaipeng/data/yelp/style_transfer
train_file=$data_dir/train.tsv
test_file=$data_dir/dev.tsv
cuda=3
expr=3
CUDA_VISIBLE_DEVICES=$cuda python run_lm_finetuning.py \
	--output_dir=lm_train/$expr/ \
	--model_type=roberta \
	--model_name_or_path=roberta-base \
	--do_train \
	--train_data_file=$train_file \
	--do_eval \
	--evaluate_during_training \
	--eval_data_file=$test_file \
	--pos_vocab $data_dir/pos_vocab.txt \
	--neg_vocab $data_dir/neg_vocab.txt \
	--mlm \
	--block_size 128 \
	--num_train_epochs 100 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--save_steps 400 \
	--logging_steps 200
