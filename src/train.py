from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def train_model(df):
    X = df.drop("pm2.5", axis=1)
    y = df["pm2.5"]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/linear_regression.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    return model, X_test, y_test

