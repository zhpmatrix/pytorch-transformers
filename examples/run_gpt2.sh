MODEL_PATH=/data/share/zhanghaipeng/data/pt_bert_models/gpt-2/large/
CUDA=2,3
CUDA_VISIBLE_DEVICES=$CUDA python run_generation.py \
	--model_type gpt2 \
	--model_name_or_path $MODEL_PATH \
	--length 15 \
	--prompt 'I like'
