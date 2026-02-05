from data_loader import load_and_clean_data
from feature_engineering import add_features
from train import train_model
from evaluate import evaluate_model

DATA_PATH = "data/raw/PRSA_data_2010.1.1-2014.12.31.csv"

df = load_and_clean_data(DATA_PATH)
df = add_features(df)

model, X_test, y_test = train_model(df)
evaluate_model(model, X_test, y_test)
