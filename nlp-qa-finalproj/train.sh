python3 main.py \
    --use_gpu \
    --model "baseline" \
    --model_path "squad_model.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --output_path "squad_predictions.txt" \
    --hidden_dim 256 \
    --bidirectional \
    --do_test \
    --device 0