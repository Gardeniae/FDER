export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
model_name=FDER

for seq_len in 192 480 720 1440 1920 2048
do
  for pred_len in 96 192 336 720
  do
    echo "========== FDER Weather seq${seq_len}_pred${pred_len} =========="
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./datasets/weather/ \
      --data_path weather.csv \
      --model_id weather_${seq_len}_${pred_len}_FDER_lr005_ep15 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --train_epochs 15 \
      --learning_rate 0.01 \
      --batch_size 32 \
      --itr 1 "$@"
  done
done
