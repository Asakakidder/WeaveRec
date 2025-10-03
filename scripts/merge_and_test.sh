adapter_list="['save/sports/checkpoint-1000','save/sports_clothing/checkpoint-4170','save/sports_beauty/checkpoint-4950']"
test_file="testdata/sports.jsonl"
output=result/c_b_to_s
func="softmax"
softmax_t=1
out=${output}/output
if [ ! -d ${output} ];then
    mkdir ${output}
fi
if [ ! -d ${out} ];then
    mkdir ${out}
fi

CUDA_VISIBLE_DEVICES=0 python result/change_weights.py \
    --weight ${adapter_list}
CUDA_VISIBLE_DEVICES=0 python src/merge_weights.py \
    --adapter_list ${adapter_list} \
    --weight_path result/merging_weights.pth \
    --func ${func} \
    --softmax_t ${softmax_t}
CUDA_VISIBLE_DEVICES=0 python src/vllmtest.py \
    --outpath ${out} \
    --outname output \
    --input ${test_file} \
    --memory 1 \
    --batch 12 \
    --max_new_tokens 800 \
    --model ${output}/tempLLM
rm -rf ${output}/tempLLM
