
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import xgboost as xgb
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# class RolloverPredictor:
#     def __init__(self):
#         self.model = None
#         self.feature_columns = None
#         self.target_column = 'rollover_weight_pct'  # Using weight-based rollover as primary target

#     def prepare_features(self, df):
#         """Prepare features for training"""
#         # Select numerical features for the model
#         feature_cols = [
#             'total_ulds', 'total_net_weight', 'total_gross_weight', 'avg_weight_per_uld',
#             'ibx_pct', 'ipf_pct', 'ieb_pct', 'ixf_pct',
#             'md_pct', 'ld_pct', 'lw_pct',
#             'no_ops_pct', 'demand_over_alloc_pct',
#             'day_of_week', 'week_of_year'
#         ]

#         # Handle categorical features with one-hot encoding
#         df_encoded = df.copy()

#         # One-hot encode origin and destination regions
#         if 'dominant_origin_region' in df.columns:
#             origin_dummies = pd.get_dummies(df['dominant_origin_region'], prefix='origin')
#             df_encoded = pd.concat([df_encoded, origin_dummies], axis=1)
#             feature_cols.extend(origin_dummies.columns.tolist())

#         if 'dominant_dest_region' in df.columns:
#             dest_dummies = pd.get_dummies(df['dominant_dest_region'], prefix='dest')
#             df_encoded = pd.concat([df_encoded, dest_dummies], axis=1)
#             feature_cols.extend(dest_dummies.columns.tolist())

#         # Store feature columns
#         self.feature_columns = [col for col in feature_cols if col in df_encoded.columns]

#         return df_encoded[self.feature_columns]

#     def train_model(self, training_data_path="rollover_training_data.csv"):
#         """Train the rollover prediction model"""
#         # Load training data
#         df = pd.read_csv(training_data_path)
#         print(f"Loaded training data with {len(df)} samples")

#         if len(df) < 3:
#             print("Warning: Very few training samples. Model may not be reliable.")

#         # Prepare features and target
#         X = self.prepare_features(df)
#         y = df[self.target_column]

#         print(f"Training with {len(self.feature_columns)} features:")
#         print(f"Features: {self.feature_columns}")

#         # Handle small dataset case
#         if len(df) < 10:
#             # Use all data for training if we have very few samples
#             X_train, X_test = X, X
#             y_train, y_test = y, y
#             print("Using all data for training due to small sample size")
#         else:
#             # Split data for training and testing
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )

#         # Try XGBoost first, fallback to RandomForest if issues
#         try:
#             # XGBoost model
#             self.model = xgb.XGBRegressor(
#                 n_estimators=100,
#                 max_depth=6,
#                 learning_rate=0.1,
#                 random_state=42,
#                 objective='reg:squarederror'
#             )
#             model_name = "XGBoost"
#         except:
#             # Fallback to Random Forest
#             self.model = RandomForestRegressor(
#                 n_estimators=100,
#                 max_depth=10,
#                 random_state=42
#             )
#             model_name = "Random Forest"

#         # Train the model
#         self.model.fit(X_train, y_train)

#         # Make predictions
#         y_pred = self.model.predict(X_test)

#         # Calculate metrics
#         mse = mean_squared_error(y_test, y_pred)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         print(f"\n{model_name} Model Performance:")
#         print(f"RMSE: {rmse:.2f}%")
#         print(f"MAE: {mae:.2f}%")
#         print(f"R² Score: {r2:.3f}")

#         # Feature importance
#         if hasattr(self.model, 'feature_importances_'):
#             feature_importance = pd.DataFrame({
#                 'feature': self.feature_columns,
#                 'importance': self.model.feature_importances_
#             }).sort_values('importance', ascending=False)

#             print("\nTop 10 Most Important Features:")
#             print(feature_importance.head(10))

#         # Cross-validation if enough samples
#         if len(df) >= 5:
#             cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(df)), scoring='neg_root_mean_squared_error')
#             print(f"\nCross-validation RMSE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

#         return self.model

#     def predict_rollover(self, current_backlog_df):
#         """Predict rollover percentage for current backlog"""
#         if self.model is None:
#             raise ValueError("Model not trained. Call train_model() first.")

#         # Calculate features from current backlog
#         features = self._calculate_features_from_backlog(current_backlog_df)

#         # Prepare feature vector
#         feature_df = pd.DataFrame([features])
#         X = self.prepare_features_for_prediction(feature_df)

#         # Make prediction
#         predicted_rollover = self.model.predict(X)[0]

#         # Ensure prediction is within reasonable bounds
#         predicted_rollover = max(0, min(100, predicted_rollover))

#         return predicted_rollover

#     def _calculate_features_from_backlog(self, backlog_df):
#         """Calculate features from current backlog DataFrame"""
#         total_ulds = len(backlog_df)

#         features = {
#             'total_ulds': total_ulds,
#             'total_net_weight': backlog_df['Net Weight (LBS)'].sum(),
#             'total_gross_weight': backlog_df['Gross Weight (LBS)'].sum(),
#             'avg_weight_per_uld': backlog_df['Net Weight (LBS)'].mean() if total_ulds > 0 else 0,
#         }

#         # Priority distribution
#         priority_counts = backlog_df['Priority'].value_counts()
#         features['ibx_pct'] = (priority_counts.get('IBX', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#         features['ipf_pct'] = (priority_counts.get('IPF', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#         features['ieb_pct'] = (priority_counts.get('IEB', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#         features['ixf_pct'] = (priority_counts.get('IXF', 0) / total_ulds) * 100 if total_ulds > 0 else 0

#         # Position distribution
#         position_counts = backlog_df['ULD Position'].value_counts()
#         features['md_pct'] = (position_counts.get('MD', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#         features['ld_pct'] = (position_counts.get('LD', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#         features['lw_pct'] = (position_counts.get('LW', 0) / total_ulds) * 100 if total_ulds > 0 else 0

#         # Reason codes
#         reason_counts = backlog_df['Reason for Backlog'].value_counts()
#         features['no_ops_pct'] = (reason_counts.get('No Ops/No network available', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#         features['demand_over_alloc_pct'] = (reason_counts.get('Demand over allocation', 0) / total_ulds) * 100 if total_ulds > 0 else 0

#         # Time features (current date)
#         current_date = datetime.now()
#         features['day_of_week'] = current_date.weekday()
#         features['week_of_year'] = current_date.isocalendar()[1]

#         # Regional features
#         origin_region_counts = backlog_df['Origin Region'].value_counts()
#         dest_region_counts = backlog_df['ULD Destination Region'].value_counts()
#         features['dominant_origin_region'] = origin_region_counts.index[0] if len(origin_region_counts) > 0 else 'Unknown'
#         features['dominant_dest_region'] = dest_region_counts.index[0] if len(dest_region_counts) > 0 else 'Unknown'

#         return features

#     def prepare_features_for_prediction(self, feature_df):
#         """Prepare features for prediction (handles one-hot encoding)"""
#         # Create dummy variables for categorical features
#         df_encoded = feature_df.copy()

#         if 'dominant_origin_region' in df_encoded.columns:
#             origin_dummies = pd.get_dummies(df_encoded['dominant_origin_region'], prefix='origin')
#             df_encoded = pd.concat([df_encoded, origin_dummies], axis=1)

#         if 'dominant_dest_region' in df_encoded.columns:
#             dest_dummies = pd.get_dummies(df_encoded['dominant_dest_region'], prefix='dest')
#             df_encoded = pd.concat([df_encoded, dest_dummies], axis=1)

#         # Ensure all training features are present (fill with 0 if missing)
#         for col in self.feature_columns:
#             if col not in df_encoded.columns:
#                 df_encoded[col] = 0

#         return df_encoded[self.feature_columns]

#     def save_model(self, filepath="rollover_model.joblib"):
#         """Save the trained model"""
#         if self.model is None:
#             raise ValueError("No model to save. Train a model first.")

#         model_data = {
#             'model': self.model,
#             'feature_columns': self.feature_columns,
#             'target_column': self.target_column
#         }

#         joblib.dump(model_data, filepath)
#         print(f"Model saved to {filepath}")

#     def load_model(self, filepath="rollover_model.joblib"):
#         """Load a trained model"""
#         try:
#             model_data = joblib.load(filepath)
#             self.model = model_data['model']
#             self.feature_columns = model_data['feature_columns']
#             self.target_column = model_data['target_column']
#             print(f"Model loaded from {filepath}")
#             return True
#         except Exception as e:
#             print(f"Error loading model: {str(e)}")
#             return False

# # Example usage and training script
# if __name__ == "__main__":
#     # Create and train the model
#     predictor = RolloverPredictor()

#     try:
#         # Train the model
#         predictor.train_model("rollover_training_data.csv")

#         # Save the model
#         predictor.save_model("rollover_model.joblib")

#         print("\n✅ Model training completed successfully!")

#     except Exception as e:
#         print(f"❌ Error during training: {str(e)}")
#         print("Make sure you have run calculate_rollover.py first to generate training data.")

#==================================================================================================================================
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import sys

# class RolloverPredictor:
#     def __init__(self, target_column="weight_rollover_rate"):
#         self.target_column = target_column
#         self.model = None
#         self.feature_names = []

#     def train_model(self, csv_path):
#         # Load data
#         df = pd.read_csv(csv_path)
#         print(f"Loaded training data with {len(df)} samples")

#         # Drop rows with missing target
#         df = df.dropna(subset=[self.target_column])
#         if len(df) == 0:
#             print("No data to train on after dropping missing targets.")
#             sys.exit(1)

#         # Features: drop non-numeric and non-feature columns
#         drop_cols = ["date", self.target_column]
#         X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
#         # Only keep numeric columns
#         X = X.select_dtypes(include=[np.number])
#         y = df[self.target_column]

#         self.feature_names = list(X.columns)
#         print(f"Features used: {self.feature_names}")

#         # Train/test split (with so little data, just for demonstration)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#         # XGBoost regressor
#         self.model = xgb.XGBRegressor(
#             n_estimators=100,
#             max_depth=3,
#             learning_rate=0.1,
#             objective="reg:squarederror",
#             random_state=42
#         )
#         self.model.fit(X_train, y_train)

#         # Evaluation
#         y_pred = self.model.predict(X_test)
#         print("Test set results:")
#         print(f"  MAE: {mean_absolute_error(y_test, y_pred):.2f}")
#         print(f"  RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
#         print(f"  R2: {r2_score(y_test, y_pred):.2f}")

#         # Cross-validation (if enough samples)
#         if len(df) >= 5:
#             scores = cross_val_score(self.model, X, y, cv=min(5, len(df)), scoring='neg_root_mean_squared_error')
#             print(f"Cross-validated RMSE: {-scores.mean():.2f}")

#         # Feature importance
#         importances = self.model.feature_importances_
#         print("Feature importances:")
#         for name, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1]):
#             print(f"  {name}: {imp:.4f}")

#         # Save model
#         joblib.dump((self.model, self.feature_names), "rollover_predictor_xgb.pkl")
#         print("Model saved as rollover_predictor_xgb.pkl")

#     def predict(self, X_new):
#         if self.model is None:
#             self.model, self.feature_names = joblib.load("rollover_predictor_xgb.pkl")
#         X_new = X_new[self.feature_names]
#         return self.model.predict(X_new)

# if __name__ == "__main__":
#     try:
#         predictor = RolloverPredictor(target_column="weight_rollover_rate")
#         predictor.train_model("rollover_training_data.csv")
#     except Exception as e:
#         # Use ascii fallback for error messages
#         print("Error during training: " + str(e))
#==============================================================================================================================

# FedEx Backlog Recovery System - ML Model Training
# Trains XGBoost model to predict optimal rollover percentages

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """Load the historical rollover training data"""
    try:
        df = pd.read_csv('rollover_training_data.csv')
        print(f" Loaded training data: {len(df)} samples")
        return df
    except FileNotFoundError:
        print(" Training data not found. Please run calculate_rollover.py first.")
        return None

def prepare_features(df):
    """Prepare features for ML model training"""
    # Select relevant features for prediction
    feature_columns = [
        'total_ulds', 'total_net_weight', 'avg_weight_per_uld',
        'pct_ibx', 'pct_ieb', 'pct_ipf', 'pct_ixf', 'pct_ief',
        'pct_md', 'pct_ld', 'pct_lw',
        'pct_am', 'pct_eu', 'pct_meisa', 'pct_as',
        'pct_dwc', 'pct_bom', 'pct_del', 'pct_blr',
        'pct_no_ops', 'pct_demand_over', 'pct_space_constraint', 'pct_planned_rollover',
        'day_of_week', 'day_of_month', 'week_of_year'
    ]
    
    # Ensure all feature columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f" Missing feature columns: {missing_cols}")
        # Add missing columns with default values
        for col in missing_cols:
            df[col] = 0.0
    
    X = df[feature_columns].fillna(0)
    
    # Target variables - we'll focus on weight-based rollover rate
    y_weight = df['weight_rollover_rate'].fillna(0)
    y_count = df['count_rollover_rate'].fillna(0)
    
    return X, y_weight, y_count, feature_columns

def train_xgboost_model(X, y, feature_names):
    """Train XGBoost regression model"""
    
    # XGBoost parameters optimized for small datasets
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,  # Shallow trees for small dataset
        'learning_rate': 0.1,
        'n_estimators': 50,  # Fewer trees to avoid overfitting
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0   # L2 regularization
    }
    
    model = xgb.XGBRegressor(**params)
    
    # For small datasets, we'll use all data for training
    # but perform cross-validation for evaluation
    model.fit(X, y)
    
    # Cross-validation for model evaluation
    cv_scores = cross_val_score(model, X, y, cv=min(3, len(X)), 
                               scoring='neg_mean_squared_error')
    
    return model, cv_scores

def evaluate_model(model, X, y, cv_scores, model_name):
    """Evaluate model performance"""
    
    # Predictions on training data (for small dataset)
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    # Cross-validation metrics
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_std = np.sqrt(-cv_scores).std()
    
    print(f"\n {model_name} Model Performance:")
    print(f"   Training RMSE: {rmse:.2f}%")
    print(f"   Training R²: {r2:.3f}")
    print(f"   CV RMSE: {cv_rmse:.2f}% (±{cv_std:.2f})")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'cv_rmse': cv_rmse,
        'cv_std': cv_std
    }

def get_feature_importance(model, feature_names):
    """Get and display feature importance"""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25}: {row['importance']:.3f}")
    
    return feature_importance

def save_model_and_metadata(weight_model, count_model, feature_names, 
                           weight_metrics, count_metrics, feature_importance):
    """Save trained models and metadata"""
    
    # Save models
    joblib.dump(weight_model, 'rollover_weight_model.pkl')
    joblib.dump(count_model, 'rollover_count_model.pkl')
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'weight_model_metrics': weight_metrics,
        'count_model_metrics': count_metrics,
        'feature_importance': feature_importance.to_dict('records'),
        'model_info': {
            'algorithm': 'XGBoost Regression',
            'target_weight': 'weight_rollover_rate (%)',
            'target_count': 'count_rollover_rate (%)',
            'training_samples': len(feature_names)
        }
    }
    
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n Models saved:")
    print(f"   - rollover_weight_model.pkl")
    print(f"   - rollover_count_model.pkl") 
    print(f"   - model_metadata.json")

# def create_sample_prediction():
#     """Create a sample prediction to test the model"""
#     try:
#         weight_model = joblib.load('rollover_weight_model.pkl')
        
#         # Load a sample from training data
#         df = pd.read_csv('rollover_training_data.csv')
#         sample = df.iloc[0:1]
        
#         # Prepare features
#         X, _, _, feature_names = prepare_features(df)
#         X_sample = X.iloc[0:1]
        
#         # Make prediction
#         prediction = weight_model.predict(X_sample)
#         actual = sample['weight_rollover_rate'].iloc
        
#         print(f"\n Sample Prediction Test:")
#         print(f"   Input: {sample['total_ulds'].iloc} ULDs, "
#               f"{sample['total_net_weight'].iloc:,.0f} lbs")
#         print(f"   Predicted rollover: {prediction:.1f}%")
#         print(f"   Actual rollover: {actual:.1f}%")
#         print(f"   Difference: {abs(prediction - actual):.1f}%")
        
#     except Exception as e:
#         print(f" Sample prediction failed: {e}")
def create_sample_prediction():
    """Create a sample prediction to test the model"""
    try:
        weight_model = joblib.load('rollover_weight_model.pkl')
        df = pd.read_csv('rollover_training_data.csv')
        X, _, _, feature_names = prepare_features(df)
        X_sample = X.iloc[0:1]
        prediction = weight_model.predict(X_sample)[0]
        actual = df['weight_rollover_rate'].iloc[0]
        total_ulds = df['total_ulds'].iloc[0]
        total_net_weight = df['total_net_weight'].iloc[0]
        print(f"\n Sample Prediction Test:")
        print(f"   Input: {total_ulds} ULDs, {total_net_weight:,.0f} lbs")
        print(f"   Predicted rollover: {prediction:.1f}%")
        print(f"   Actual rollover: {actual:.1f}%")
        print(f"   Difference: {abs(prediction - actual):.1f}%")
    except Exception as e:
        print(f" Sample prediction failed: {e}")


def main():
    """Main training workflow"""
    print("=== FedEx Backlog ML Model Training ===")
    print("Training XGBoost models for rollover prediction...\n")
    
    # Load data
    df = load_training_data()
    if df is None:
        return
    
    # Prepare features
    X, y_weight, y_count, feature_names = prepare_features(df)
    
    print(f" Training Features: {len(feature_names)} features")
    print(f" Training Samples: {len(X)} days")
    print(f" Weight rollover range: {y_weight.min():.1f}% - {y_weight.max():.1f}%")
    
    # Train models
    print(f"\n🔬 Training models...")
    weight_model, weight_cv = train_xgboost_model(X, y_weight, feature_names)
    count_model, count_cv = train_xgboost_model(X, y_count, feature_names)
    
    # Evaluate models thishas to be done on a daily basis
    weight_metrics = evaluate_model(weight_model, X, y_weight, weight_cv, "Weight Rollover")
    count_metrics = evaluate_model(count_model, X, y_count, count_cv, "Count Rollover")
    
    # Feature importance
    feature_importance = get_feature_importance(weight_model, feature_names)
    
    # Save everything
    save_model_and_metadata(weight_model, count_model, feature_names,
                           weight_metrics, count_metrics, feature_importance)
    
    # Test prediction
    create_sample_prediction()
    
    print(f"\n Training complete! Models ready for use in Streamlit app.")
    print(f"   Run: streamlit run app_with_ml.py")


# (newly added on 07/07/25)
class RolloverPredictor:
    def __init__(self, model_path='rollover_weight_model.pkl', feature_path='model_metadata.json'):
        self.model = joblib.load(model_path)
        import json
        with open(feature_path, 'r') as f:
            meta = json.load(f)
        self.feature_names = meta['feature_names']

    def predict(self, df_features):
        # Ensure columns match training order
        X = df_features[self.feature_names].fillna(0)
        return self.model.predict(X)
    
if __name__ == "__main__":
    main()
#===============================================================================================================================