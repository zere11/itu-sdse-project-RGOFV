import os
import warnings
import pandas as pd

from data.dataset import data_extraction
from data.preprocess import data_preparation

def main():
    os.makedirs("./artifacts", exist_ok=True)
    warnings.filterwarnings('ignore')
    pd.set_option('display.float_format', lambda x: "%.3f" % x)

    df_data = data_extraction()
    df_data = data_preparation(df_data)
    df_data.to_csv('./artifacts/train_data_gold.csv', index=False)
    print("Data extraction completed.")

if __name__ == "__main__":
    main()
