from src.load_data import load_data
from src.preprocess import preprocess_data
from src.train_model import train_and_evaluate

def main():
    print("🏠 Loading dataset...")
    df = load_data("data/data.csv")

    print("\n🧹 Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\n🤖 Training model...")
    model, mse, r2 = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n📊 Model Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

if __name__ == "__main__":
    main()
