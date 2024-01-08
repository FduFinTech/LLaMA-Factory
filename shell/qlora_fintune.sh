deepspeed --num_gpus 4 --master_port=9901 src/train_bash.py \
    --deepspeed ds_configs/ds_config_zero2.json \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen-7B \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target c_attn \
    --quantization_bit 4 \
    --output_dir model_output \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16