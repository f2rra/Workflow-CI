import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss, jaccard_score
import datetime

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments for MLflow Project"""
    parser = argparse.ArgumentParser(description='Weather Prediction Model Training')
    
    # Data parameters
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Test set size ratio (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    
    # Model parameters
    parser.add_argument('--n_estimators', type=int, default=None,
                        help='Number of estimators (if not specified, will use grid search)')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Maximum depth (if not specified, will use grid search)')
    parser.add_argument('--min_samples_split', type=int, default=None,
                        help='Minimum samples split (if not specified, will use grid search)')
    parser.add_argument('--min_samples_leaf', type=int, default=None,
                        help='Minimum samples leaf (if not specified, will use grid search)')
    
    # Training parameters
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='Number of cross-validation folds (default: 3)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1)')
    parser.add_argument('--experiment_name', type=str, default='Weather_Prediction',
                        help='MLflow experiment name (default: Weather_Prediction)')
    
    return parser.parse_args()

def setup_mlflow():
    """Setup MLflow tracking configuration"""
    # Get credentials
    username = os.getenv('MLFLOW_TRACKING_USERNAME')
    password = os.getenv('MLFLOW_TRACKING_PASSWORD')

    if not username or not password:
        raise ValueError("DagsHub credentials not found. Please create a .env file with MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD")

    # Configure DagsHub
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = password

def load_and_prepare_data():
    """Load and prepare data for training"""
    print("Loading preprocessed data...")
    
    # Check if data file exists
    data_file = 'weather_preprocessed.csv'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found. Please ensure the file exists in the MLProject directory.")
    
    data = pd.read_csv(data_file)
    print(f"Data loaded successfully. Shape: {data.shape}")
    
    # Target columns
    target_columns = [
        'weather_description_Berawan',
        'weather_description_Cerah',
        'weather_description_Cerah Berawan',
        'weather_description_Hujan Ringan',
        'weather_description_Hujan Sedang'
    ]
    
    # Verify target columns exist
    missing_targets = [col for col in target_columns if col not in data.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    
    # Drop 'month' column if all values are 0
    if 'month' in data.columns and (data['month'] == 0).all():
        data = data.drop('month', axis=1)
        print("Dropped 'month' column (all zeros)")
    
    # Split features and targets
    X = data.drop(target_columns, axis=1)
    y = data[target_columns]
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y, target_columns

def create_model(args):
    """Create model with hyperparameters"""
    # Check if specific parameters are provided
    if all(param is not None for param in [args.n_estimators, args.max_depth, 
                                          args.min_samples_split, args.min_samples_leaf]):
        # Use specific parameters
        print("Using provided hyperparameters...")
        base_model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            class_weight='balanced'
        )
        model = MultiOutputClassifier(base_model)
        return model, None
    else:
        # Use grid search
        print("Using grid search for hyperparameter tuning...")
        param_grid = {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__max_depth': [None, 10, 20, 30],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__bootstrap': [True, False]
        }
        
        base_model = RandomForestClassifier(random_state=args.random_state, class_weight='balanced')
        model = MultiOutputClassifier(base_model)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=args.cv_folds,
            scoring='f1_micro',
            verbose=1,
            n_jobs=args.n_jobs
        )
        
        return model, grid_search

def train_model(model, grid_search, X_train, y_train, args):
    """Train the model"""
    print("Training model...")
    
    if grid_search is not None:
        # Grid search training
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        return best_model, best_params
    else:
        # Direct training
        model.fit(X_train, y_train)
        best_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf
        }
        return model, best_params

def evaluate_model(model, X_test, y_test, target_columns):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1_micro = f1_score(y_test, predictions, average='micro')
    f1_macro = f1_score(y_test, predictions, average='macro')
    hamming = hamming_loss(y_test, predictions)
    jaccard_micro = jaccard_score(y_test, predictions, average='micro')
    
    metrics = {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'hamming_loss': hamming,
        'jaccard_micro': jaccard_micro
    }
    
    print(f"Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Micro: {f1_micro:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Jaccard Micro: {jaccard_micro:.4f}")
    
    # Classification report
    report = classification_report(y_test, predictions, target_names=target_columns)
    
    return metrics, predictions, report

def save_artifacts(model, X, metrics, report, best_params):
    """Save training artifacts"""
    print("Saving artifacts...")
    
    # Save classification report
    with open("classification_report.txt", "w") as f:
        f.write(report)
    
    # Save best parameters
    with open("best_parameters.txt", "w") as f:
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Metrics: {metrics}\n")
    
    # Feature importance
    if hasattr(model.estimators_[0], 'feature_importances_'):
        importance = model.estimators_[0].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("feature_importance.csv", index=False)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved feature importance for {len(feature_importance)} features")

def main():
    try:
        # Parse arguments
        args = parse_args()
        print(f"Starting training with arguments: {vars(args)}")
        
        # Setup MLflow
        setup_mlflow()
        mlflow.set_experiment(args.experiment_name)
        
        # Load data
        X, y, target_columns = load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        
        # Enable MLflow autolog
        mlflow.sklearn.autolog()
        
        with mlflow.start_run():
            # Set custom tags
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("problem_type", "Multi-label Classification")
            mlflow.set_tag("data_version", "v1.0")
            mlflow.set_tag("training_mode", "MLflow_Project")
            
            # Create and train model
            model, grid_search = create_model(args)
            best_model, best_params = train_model(model, grid_search, X_train, y_train, args)
            
            # Evaluate model
            metrics, predictions, report = evaluate_model(best_model, X_test, y_test, target_columns)
            
            # Log parameters and metrics
            mlflow.log_params(best_params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Additional info
            mlflow.log_metric("num_features", len(X.columns))
            mlflow.log_metric("train_samples", X_train.shape[0])
            mlflow.log_metric("test_samples", X_test.shape[0])
            
            # Save artifacts
            save_artifacts(best_model, X, metrics, report, best_params)
            
            # Log artifacts
            mlflow.log_artifact("classification_report.txt")
            mlflow.log_artifact("best_parameters.txt")
            if os.path.exists("feature_importance.csv"):
                mlflow.log_artifact("feature_importance.csv")
            if os.path.exists("feature_importance.png"):
                mlflow.log_artifact("feature_importance.png")
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="weather_model",
                registered_model_name="WeatherPredictor"
            )
            
            print(f"\n✅ Training completed successfully!")
            print(f"📊 MLflow UI: https://dagshub.com/f2rra/Eksperimen_SML_Fathur.mlflow")
            print(f"🎯 Best F1 Micro Score: {metrics['f1_micro']:.4f}")
            
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()