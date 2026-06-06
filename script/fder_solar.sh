export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

model_name=FDER
for seq_len in 192 480 720 1440 1920 2048
do
  for pred_len in 96 192 336 720
  do
    echo "========== FDER Solar seq${seq_len}_pred${pred_len} =========="
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./datasets/Solar/ \
      --data_path solar_AL.txt \
      --model_id solar_${seq_len}_${pred_len}_FDER_lr005_ep15 \
      --model $model_name \
      --data Solar \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --train_epochs 15 \
      --learning_rate 0.005 \
      --batch_size 32 \
      --itr 1 "$@"
  done 
done
