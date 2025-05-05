import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Drop non-numeric / non-useful columns
    df = df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1)

    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
