export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
model_name=FDER
seq_len=480

for pred_len in 96 192 336 720
do
  echo "========== FDER Exchange seq${seq_len}_pred${pred_len} =========="
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_${seq_len}_${pred_len}_FDER_lr005_ep15 \
    --model $model_name \
    --data custom \
    --features M \
    --freq d \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --train_epochs 15 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --itr 1 "$@"
done
