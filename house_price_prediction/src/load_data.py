import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("📊 Dataset Loaded:")
    print(df.head())
    print("\n🧾 Columns:", df.columns.tolist())
    print("\n🧼 Missing Values:\n", df.isnull().sum())
    return df
