export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=FDER
# 192 480 720 1440 1920  
seq_len=1920
for pred_len in 96 192 336 720
do
  echo "========== FDER ETTm2 seq${seq_len}_pred${pred_len} =========="
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_${seq_len}_${pred_len}_FDER_lr005_ep15 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --train_epochs 15 \
    --learning_rate 0.005 \
    --batch_size 32 \
    --itr 1 "$@"
done
