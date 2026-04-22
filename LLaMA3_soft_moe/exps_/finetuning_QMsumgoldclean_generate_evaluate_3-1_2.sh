# export CUDA_VISIBLE_DEVICES="2"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

max_devices=2

if [ "$num_devices" -gt "$max_devices" ]; then
    num_devices=$max_devices
    echo "max of devices: $max_devices"
fi

# train
epochs=3
dataset="QMSum_gold_clean"
max_seq_len=3000
min_gen_len=120
lora_rank=4
lora_targets="Q,K,V,O"
lora_alpha=64
expert_num=6
hydra_moe=True # hydra lora, Asymmetric LoRA
blr=6e-3
flash_attention2=True
bf16=True
tag=""
path="/home2/caojie"
output_dir="${path}/outputs/LLaMA3-1_lora_moe_s/${dataset}/b32_epoch${epochs}_warme1_lorar${lora_rank}_lora${lora_targets}_alpha${lora_alpha}_expertnum${expert_num}_hydra${hydra_moe}_blr${blr}_maxseq${max_seq_len}_flashatt2${flash_attention2}_bf16${bf16}_${tag}/"

torchrun --nproc_per_node $num_devices --master_port=3037 main_finetune.py \
    --llama_path ${path}/pretrain_models/Meta-Llama-3.1-8B-Instruct/ \
    --data_path ${path}/datasets/${dataset}/train.jsonl \
    --val_data_path ${path}/datasets/${dataset}/test.jsonl \
    --lora_rank ${lora_rank} \
    --w_lora True \
    --lora_targets $lora_targets \
    --lora_alpha $lora_alpha \
    --expert_num $expert_num \
    --hydra_moe $hydra_moe \
    --max_seq_len $max_seq_len \
    --batch_size 1 \
    --accum_iter $((32/$num_devices)) \
    --epochs ${epochs} \
    --warmup_epochs 1 \
    --blr ${blr} \
    --flash_attention2 $flash_attention2 \
    --bf16 $bf16 \
    --weight_decay 0.02 \
    --output_dir $output_dir \
    --num_workers 10

checkpoint="${output_dir}checkpoint-$((epochs-1)).pth"
# get lora parameters
python extract_adapter_from_checkpoint.py --checkpoint $checkpoint

adapter_path="${output_dir}adapter.pth"
save_path="${output_dir}predict_mingen$min_gen_len.jsonl"
torchrun --nproc_per_node $num_devices --master_port=3037 example.py \
    --ckpt_dir ${path}/pretrain_models/Meta-Llama-3.1-8B-Instruct/ \
    --adapter_path $adapter_path \
    --data_path ${path}/datasets/${dataset}/test.jsonl \
    --save_path $save_path \
    --max_gen_len 128 \
    --min_gen_len $min_gen_len \
    --max_batch_size 10 \
    --temperature 0.1 \
    --top_p 0.75

bscore_path="${path}/pretrain_models/bart_base"
python evaluate.py --predict_file $save_path --bscore_path $bscore_path