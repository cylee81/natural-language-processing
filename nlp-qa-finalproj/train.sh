python3 main.py \
    --use_gpu \
    --model "baseline" \
    --model_path "Elmo-1024_1.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --output_path "Elmo-1024_1.txt" \
    --hidden_dim 256 \
	--embedding_dim 1024 \
	--batch_size 32 \
    --bidirectional \
    --do_train \
    --do_test \
    --device 0

python3 main.py \
    --use_gpu \
    --model "baseline" \
    --model_path "Elmo-1024_2.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --output_path "Elmo-1024_2.txt" \
    --hidden_dim 256 \
	--batch_size 32 \
    --bidirectional \
    --do_train \
    --do_test \
    --device 0

python3 main.py \
    --use_gpu \
    --model "baseline" \
    --model_path "only_ner_3.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --output_path "only_ner_3.txt" \
    --hidden_dim 256 \
    --bidirectional \
    --do_train \
    --do_test \
    --device 0
