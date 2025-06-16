import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss, jaccard_score
import datetime

# Load environment variables
load_dotenv()

# Get credentials
username = os.getenv('MLFLOW_TRACKING_USERNAME')
password = os.getenv('MLFLOW_TRACKING_PASSWORD')

if not username or not password:
    raise ValueError("Dagshub credentials not found. Please create a .env file with MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD")

# Konfigurasi DagsHub
os.environ['MLFLOW_TRACKING_USERNAME'] = username
os.environ['MLFLOW_TRACKING_PASSWORD'] = password

# Set MLflow tracking URI (DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/f2rra/Eksperimen_SML_Fathur.mlflow")
mlflow.set_experiment("Weather_Prediction")

def main():
    # Load preprocessed data
    data = pd.read_csv('weather_preprocessed.csv')
    
    # Siapkan target: 5 kolom weather_description
    target_columns = [
        'weather_description_Berawan',
        'weather_description_Cerah',
        'weather_description_Cerah Berawan',
        'weather_description_Hujan Ringan',
        'weather_description_Hujan Sedang'
    ]
    
    # Drop kolom 'month' jika semua nilainya 0
    if 'month' in data.columns and (data['month'] == 0).all():
        data = data.drop('month', axis=1)
    
    # Split data
    X = data.drop(target_columns, axis=1)
    y = data[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Aktifkan autolog MLflow
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # Set custom tags
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("problem_type", "Multi-label Classification")
        mlflow.set_tag("data_version", "v1.0")
        
        # Hyperparameter tuning untuk klasifikasi multi-label
        param_grid = {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__max_depth': [None, 10, 20, 30],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__bootstrap': [True, False]
        }
        
        # Gunakan MultiOutputClassifier untuk multi-label classification
        base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        model = MultiOutputClassifier(base_model)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='f1_micro',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate
        predictions = best_model.predict(X_test)
        
        # Hitung metrik standar
        accuracy = accuracy_score(y_test, predictions)
        f1_micro = f1_score(y_test, predictions, average='micro')
        f1_macro = f1_score(y_test, predictions, average='macro')
        
        # Hitung metrik tambahan di luar yang biasanya dicover autolog
        hamming = hamming_loss(y_test, predictions)
        jaccard_micro = jaccard_score(y_test, predictions, average='micro')
        
        # Manual logging
        mlflow.log_params(grid_search.best_params_)
        
        # Log metrik standar
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_micro", f1_micro)
        mlflow.log_metric("f1_macro", f1_macro)
        
        # Log metrik tambahan
        mlflow.log_metric("hamming_loss", hamming)
        mlflow.log_metric("jaccard_micro", jaccard_micro)
        
        # Classification report
        report = classification_report(y_test, predictions, target_names=target_columns)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        # Log model dengan nama spesifik
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="weather_model",
            registered_model_name="WeatherPredictor"
        )
        
        # Log feature importance
        if hasattr(best_model.estimators_[0], 'feature_importances_'):
            importance = best_model.estimators_[0].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            # Visualisasi feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
        
        # Log informasi tambahan
        mlflow.log_text(f"Best parameters: {grid_search.best_params_}", "best_parameters.txt")
        mlflow.log_metric("num_features", len(X.columns))
        mlflow.log_metric("train_samples", X_train.shape[0])
        mlflow.log_metric("test_samples", X_test.shape[0])
        
        print(f"Best model trained! Accuracy: {accuracy:.4f}, F1 Micro: {f1_micro:.4f}")
        print(f"Additional metrics - Hamming Loss: {hamming:.4f}, Jaccard Micro: {jaccard_micro:.4f}")
        print(f"View run at: https://dagshub.com/f2rra/Eksperimen_SML_Fathur.mlflow")

if __name__ == "__main__":
    main()