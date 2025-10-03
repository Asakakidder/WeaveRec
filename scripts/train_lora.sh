dataset=$1
output_model=./save/${dataset}
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path models/Qwen2-7B-Instruct \
    --dataset ${dataset} \
    --dataset_dir ./traindata \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_dropout 0.05 \
    --output_dir ${output_model} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1200 \
    --preprocessing_num_workers 96 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --save_steps 30 \
    --eval_steps 30 \
    --save_total_limit 150 \
    --seed 42 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --val_size 0.02 \
    --plot_loss \
    --bf16 \
    --report_to "tensorboard" \
    | tee -a ${output_model}/${dataset}2.log