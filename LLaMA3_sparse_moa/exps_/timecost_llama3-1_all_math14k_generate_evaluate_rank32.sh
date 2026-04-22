export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

max_devices=2

if [ "$num_devices" -gt "$max_devices" ]; then
    num_devices=$max_devices
    echo "max of devices: $max_devices"
fi

# train
epochs=2
warmup_epochs=1
dataset="math_14k"
max_seq_len=300
min_gen_len=120
max_gen_len=200

lora_layers="0-32"
lora_rank=32
lora_targets="Q,K,V,O,FFN_UP,FFN_GATE,FFN_DOWN"
lora_alpha=32
bool_weights=False
max_threshold=0.5
adapter_noisy=False
const_threshold=False

p_adapter_layers="0-32"
p_adapter_size=64

prompt_layers="0-0"
prompt_len=10

swi_x=4

blr=6e-3
flash_attention2=False
bf16=True
tag=""
batch_size_gpu=4
eff_batch_size=32
path="/home2/caojie"
# output_dir="${path}/outputs/LLaMA3-1_smoe_structure/${dataset}/b${eff_batch_size}_gpu${num_devices}_bsg${batch_size_gpu}_ep${epochs}_wep${warmup_epochs}_loral${lora_layers}_lorar${lora_rank}_lora${lora_targets}_alpha${lora_alpha}_maxth${max_threshold}_constW${const_threshold}_noisy${adapter_noisy}_palayers${p_adapter_layers}_pasize${p_adapter_size}_promptl${prompt_layers}_promptl${prompt_len}_swi_x${swi_x}_blr${blr}_maxseq${max_seq_len}_flashatt2${flash_attention2}_bf16${bf16}_${tag}/"
output_dir="${path}/outputs/LLaMA3-1_smoe_structure/${dataset}/b${eff_batch_size}_gpu2_bsg${batch_size_gpu}_ep${epochs}_wep${warmup_epochs}_loral${lora_layers}_lorar${lora_rank}_lora${lora_targets}_alpha${lora_alpha}_maxth${max_threshold}_constW${const_threshold}_noisy${adapter_noisy}_palayers${p_adapter_layers}_pasize${p_adapter_size}_promptl${prompt_layers}_promptl${prompt_len}_swi_x${swi_x}_blr${blr}_maxseq${max_seq_len}_flashatt2${flash_attention2}_bf16${bf16}_${tag}/"

# torchrun --nproc_per_node $num_devices --master_port=3038 main_finetune.py \
#     --llama_path ${path}/pretrain_models/Meta-Llama-3.1-8B-Instruct/ \
#     --data_path ${path}/datasets/${dataset}/train.json \
#     --max_threshold $max_threshold \
#     --bool_weights $bool_weights \
#     --adapter_noisy $adapter_noisy \
#     --const_threshold $const_threshold \
#     --lora_layers $lora_layers \
#     --lora_rank ${lora_rank} \
#     --lora_targets $lora_targets \
#     --lora_alpha $lora_alpha \
#     --p_adapter_layers $p_adapter_layers \
#     --p_adapter_size $p_adapter_size \
#     --prompt_layers $prompt_layers\
#     --prompt_len $prompt_len \
#     --swi_x $swi_x \
#     --max_seq_len $max_seq_len \
#     --batch_size  $batch_size_gpu \
#     --accum_iter $(($eff_batch_size/$num_devices/$batch_size_gpu)) \
#     --epochs ${epochs} \
#     --warmup_epochs $warmup_epochs \
#     --blr ${blr} \
#     --flash_attention2 $flash_attention2 \
#     --bf16 $bf16 \
#     --weight_decay 0.02 \
#     --output_dir $output_dir \
#     --num_workers 10

# checkpoint="${output_dir}checkpoint-$((epochs-1)).pth"
# # get lora parameters
# python extract_adapter_from_checkpoint.py --checkpoint $checkpoint

adapter_path="${output_dir}adapter.pth"


# test_dataset_l="AddSub AQuA gsm8k MultiArith SingleEq SVAMP"
test_dataset_l="AddSub gsm8k"

for test_dataset in $test_dataset_l
do
save_path="${output_dir}${test_dataset}_mingen${min_gen_len}_getTime.jsonl"
torchrun --nproc_per_node $num_devices --master_port=3138 example_time.py \
    --ckpt_dir ${path}/pretrain_models/Meta-Llama-3.1-8B-Instruct/ \
    --adapter_path $adapter_path \
    --data_path ${path}/datasets/math_commonsense/${test_dataset}/test.json \
    --save_path $save_path \
    --max_gen_len $max_gen_len \
    --min_gen_len $min_gen_len \
    --max_batch_size 300 \
    --time_gen True \
    --temperature 0.1 \
    --top_p 0.75
done

# save_path1="${output_dir}AddSub_predict_mingen${min_gen_len}.jsonl"
# python evaluate_math.py --predict_file $save_path1