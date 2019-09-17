TAG=test
NAME=run_train.sh
GPU_NUM=1
yudctl run -p /data/share/zhanghaipeng/spellchecker/pytorch-transformers/examples -g $GPU_NUM -m 40 --gpu_family 20 -r requirements.txt -t $TAG -i registry.cn-hangzhou.aliyuncs.com/eigenlab/yudexcutor:pytorch1.0 sh $NAME
