#cd Context-Memory
#
#CUDA_VISIBLE_DEVICES=2 python run.py \
#    --model llama-3.1-8b-instruct \
#    --dataset gsm8k \
#    --train \
#    --run_id 20251216_comp_OURS

export PYTHONPATH=:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0 python src/analysis_module/eval_compression.py \
  --do_baseline
#    --model outputs/baseline_insert_COMP_after_newline-llama-3.1-8b-instruct-online-concat_recur
