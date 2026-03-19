# Mental Health Scoring (MLOps Assignment)

## Name
Name

## Roll number
123

## Dataset
Student Mental Health (Kaggle): https://www.kaggle.com/datasets/shariful07/student-mental-health

## How to run
1. Download the Kaggle CSV and place it in `data/raw/` (any `.csv` name), or set `DATA_PATH` to the CSV path.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## MLflow UI
```bash
mlflow ui
```

## Part A (training.py)
```bash
python training.py --experiment_name SKCT_727823TUAM009_MentalHealth
```

## Part B (pipeline scripts)
```bash
python data_prep.py --roll_no 123
python train_pipeline.py --roll_no 123 --experiment_name SKCT_727823TUAM009_MentalHealth
python evaluate.py --roll_no 123
```

## Azure Pipeline
Use `pipeline_123.yml`.
