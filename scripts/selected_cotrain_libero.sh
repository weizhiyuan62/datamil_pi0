set -e
RUN_TIME=$(date "+%Y-%m-%d/%H-%M-%S")
EXP_DIR=/mnt/home/weizhiyuan/data/research_wzy/datamil/exp/2026-05-27/00-50-20
VAL_NAME=study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
TASK_SHORTHAND=book-caddy

PRIOR_NAME=libero90
OCTO_PATH="/mnt/home/weizhiyuan/data/research_wzy/datamil/thirdparty/model_ckpts/octo/octo_small"

save_dir=${EXP_DIR}/policy_learning/${TASK_SHORTHAND}
idx_path=${EXP_DIR}/datamodels/${PRIOR_NAME}/${TASK_SHORTHAND}/selected_indices_topk0.1.npy

for seed in 0 1 2 3 4; do

    python scripts/cotraining.py \
        --config scripts/configs/cotrain_libero_config.py:${VAL_NAME},${PRIOR_NAME},${idx_path} \
        --config.action_chunks=8 \
        --config.seed $seed \
        --config.save_dir $save_dir  \
        --config.pretrained_path=$OCTO_PATH \
        --name seed${seed}_${PRIOR_NAME}_${TASK_SHORTHAND} \
        --debug true

done

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa python datamil/eval_libero_parallel.py \
    --task_name ${TASK_SHORTHAND} \
    --ckpt ${save_dir}/octo_finetune \
    --out_dir ${EXP_DIR}/eval_results/${TASK_SHORTHAND}