# e5-el
Entity Linking using E5


## Usage

### Instllation
```
git clone git@github.com:YuSawan/e5-el.git
cd e5_el
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Dataset preparation
#### Dataset
```
{
  "id": "doc-001",
  "examples": [
    {
      "paragraph-id": "doc-001-P1",
      "text": "She graduated from NAIST.",
      "entities": [
        {
          "start": 19,
          "end": 24,
          "label": ["000011"]
        }
      ],
    }
  ]
}
```

#### Dictionary
```
{
  "id": "000011",
  "name": "NAIST",
  "description": "NAIST is located in Ikoma."
}
```

### Finetuning

#### 1st stage
```
python src/main.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file data/train.jsonl \
    --validation_file data/dev.jsonl \
    --test_file data/test.jsonl \
    --dictionary_file data/dicttionary.jsonl \
    --model_name intfloat/multilingual-e5-large\
    --measure cos \
    --negative inbatch \
    --top_k 10 \
    --add_nil False \
    --output_dir ./initial_output/ \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 20 \
    --num_train_epochs 4
```

#### 2nd stage
```
python src/main.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file data/train.jsonl \
    --validation_file data/dev.jsonl \
    --test_file data/test.jsonl \
    --dictionary_file data/dicttionary.jsonl \
    --model_name ./initial_output/ \
    --measure cos \
    --negative inbatch \
    --top_k 10 \
    --add_nil False \
    --output_dir ./second_output/ \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 20 \
    --num_train_epochs 4
```

### Evaluation/Prediction
```
python src/main.py \
    --do_eval \
    --do_predict \
    --train_file data/train.jsonl \
    --validation_file data/dev.jsonl \
    --test_file data/test.jsonl \
    --dictionary_file data/dicttionary.jsonl \
    --model_name PATH_TO_YOUR_MODEL \
    --measure cos \
    --add_nil False \
    --output_dir PATH_TO_YOUR_MODEL \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 20 \
    --num_train_epochs 4
```
