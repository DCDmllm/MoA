export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=llama-3-1-8b-instruct
model=softmoa
dataset=codealpaca20k

# python train.py @configs/${dataset}/${base_model}_${model}_train.config

python test_code10.py @configs/${dataset}/${base_model}_${model}_test.config

python evaluate_code.py --predict_file outputs/${base_model}-${model}-${dataset}/predictions/humaneval_responses.jsonl
