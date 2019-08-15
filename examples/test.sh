DATA_PATH=/data/share/zhanghaipeng/data/chuangtouribao/event/
LOG_PATH=/data/share/zhanghaipeng/data/chuangtouribao/event/train/event
MODEL_PATH=/data/share/zhanghaipeng/data/pt_bert_models

CUDA=0,1,2,3
BATCH=32
EPOCH=3
LR=3e-5
SAVE_STEPS=100
EXPR=0
CKPT_STEP=200
CUDA_VISIBLE_DEVICES=$CUDA python run_event.py \
	--data_dir $DATA_PATH \
	--model_type bert \
	--model_name_or_path $MODEL_PATH/bert-base-chinese \
	--task_name event \
	--output_dir $LOG_PATH/$EXPR/checkpoint-$CKPT_STEP \
	--do_eval \
	--do_lower_case \
	--learning_rate $LR \
	--per_gpu_train_batch_size $BATCH \
	--num_train_epochs $EPOCH \
	--save_steps $SAVE_STEPS
	
