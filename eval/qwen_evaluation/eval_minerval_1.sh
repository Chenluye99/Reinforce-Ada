dataset_base_path='/home/chenluy/reinforce_flow/data/gen_data/qwen2.5math-1.5b-gen8-global-meanvar-nostd-iter'

for step in 400; do
    echo "global_step_$step"
    python compute_score_minerval.py \
        --dataset_path $dataset_base_path/global_step_$step/weqweasdas/minerva_math/merged_data.jsonl \
        --record_path $dataset_base_path/global_step_$step/weqweasdas/minerva_math/new_record.txt
done
