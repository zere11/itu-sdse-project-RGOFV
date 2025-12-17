# ITU BDS MLOPS'25 - Project

# ITU SDSE Project - Lead Model Pipeline (Naive Implementation)

A simplified, manual implementation of a machine learning pipeline using Dagger and Go. This is a naive approach without DVC - all data must be present locally before running.

## Folder Structure

├── source/ # Pipeline scripts                                                       
│ ├── makedataset.py # Load raw data                                             
│ ├── preprocess.py # Clean and prepare data                                             
│ ├── features.py # Feature engineering                                             
│ ├── train.py # Train models with MLflow                                             
│ ├── helpers.py # Utility functions                                             
│ ├── wrappers.py # MLflow model wrappers                                             
│ └── dataset.py # Data loading                                             
│                                             
├── artifacts/ # Input data & outputs                                             
│ ├── raw_data.csv # Input data (must exist)                                             
│ ├── train_data_gold.csv # Processed data                                             
│ ├── model.pkl # Best model                                             
│ └── '...'.json, '...'.csv # Metrics and artifacts                                             
│                                             
├── pipeline.go # Dagger pipeline definition                                             
├── requirements.txt # Python dependencies                                             
└── README.md # This file                                             
                                             
This is a **naive implementation**:                                             

No DVC integration - data must be manually managed
Manual cookiecutter-style pipeline (not fully automated)
Minimal error handling
mlflow not initiated
