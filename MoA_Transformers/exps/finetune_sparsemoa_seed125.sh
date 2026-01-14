export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=qwen3-8b
model=sparsemoa
seed=seed125

python train.py @configs/${base_model}_${model}_math14k_train_${seed}.config

python test_math.py @configs/${base_model}_${model}_math14k_test_${seed}.config

python evaluate_math.py --predict_file /data/workspace/projects/moe/MoA_Transformers/outputs/${base_model}-${model}-${seed}-math-14k/predictions/addsub_responses.jsonl
