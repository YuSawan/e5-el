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
python src/cli/run.py \
    --do_train \
    --config_file configs/config.yaml \
    --output_dir ./initial_output/
```

#### 2nd stage
```
python src/cli/run.py \
    --do_train \
    --config_file configs/config.yaml \
    --negative dense \
    --prev_path ./initial_output/ \
    --output_dir ./second_output/
```

### Evaluation/Prediction
```
python src/cli/run.py \
    --do_eval \
    --do_predict \
    --config_file configs/config.yaml
    --prev_path PATH_TO_YOUR_MODEL
    --output_dir PATH_TO_YOUR_MODEL
```
