import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys
import dagshub

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_preprocessed.csv")
    data = pd.read_csv(file_path)

    # Siapkan target: 5 kolom weather_description
    target_columns = [
        'weather_description_Berawan',
        'weather_description_Cerah',
        'weather_description_Cerah Berawan',
        'weather_description_Hujan Ringan',
        'weather_description_Hujan Sedang'
    ]
    
    # Split data
    X = data.drop(target_columns, axis=1)
    y = data[target_columns].idxmax(axis=1)
    y = y.str.replace('weather_description_', '')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y
    )

    input_example = X_train.head()
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Evaluasi
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        model_classes = model.classes_


        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
        )
        model.fit(X_train, y_train)
        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)