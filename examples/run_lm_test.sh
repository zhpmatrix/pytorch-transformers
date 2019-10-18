data_dir=/data/share/zhanghaipeng/data/yelp/style_transfer
train_file=$data_dir/train.tsv
test_file=$data_dir/dev.tsv
cuda=3
expr=2
ckpt=142400
CUDA_VISIBLE_DEVICES=$cuda python run_lm_finetuning.py \
	--output_dir=lm_train/$expr/checkpoint-$ckpt \
	--model_type=roberta \
	--model_name_or_path=roberta-base \
	--train_data_file=$train_file \
	--pos_vocab $data_dir/pos_vocab.txt \
	--neg_vocab $data_dir/neg_vocab.txt \
	--do_eval \
	--eval_data_file=$test_file \
	--mlm \
	--block_size 128 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--save_steps 500 \
	--logging_steps 50
