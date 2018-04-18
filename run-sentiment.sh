#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M aparajitah@gmail.com
#$ -m eas
#$ -l 'gpu=1,hostname=b1[123456789]*|c*,mem_free=2.1g,ram_free=2.1g'
#$ -e /export/c02/ahaldar/recast-experiments/InferSent/Sentiment/out.err
#$ -o /export/c02/ahaldar/recast-experiments/InferSent/Sentiment/out.qsub.log

source /home/apoliak/.bashrc
source activate mtie

#VARS
NLI_PATH=/export/a13/ahaldar/InferSent-backup/dataset/Sentiment/
OUTPUT_PATH=/export/c02/ahaldar/recast-experiments/InferSent/Sentiment/
SRC_PATH=/export/a13/ahaldar/InferSent-backup/output-mnli-3way/

device=0
echo $device
time CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$device python train_nli.py \
--nlipath $NLI_PATH \
--outputdir ${OUTPUT_PATH}infersent_mnli_update/ \
--pre_trained_model ${SRC_PATH}model.pickle \
--gpu_id $device

echo -e "\n\n--------- \n\nEVALUATION TIME \n\n---------\n\n"

time CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$device python eval.py \
--nlipath $NLI_PATH \
--outputdir ${OUTPUT_PATH}infersent_mnli_update/eval/ \
--model ${OUTPUT_PATH}infersent_mnli_update/model.pickle \
--gpu_id $device

