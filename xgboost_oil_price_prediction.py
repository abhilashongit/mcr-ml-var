#!/usr/bin/env python3
"""
XGBoost Model Implementation for Oil Price Prediction

This script implements a comprehensive XGBoost machine learning model 
to predict oil prices using economic indicators and supply/demand factors.

Author: Generated for Oil Price XGBoost Analysis
Date: September 28, 2025
Version: 1.0

Requirements:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- plotly (optional for interactive plots)

Input Files Expected:
- crude-oil-eu.xlsx (EU oil prices with 'Monthly' sheet)
- crude-oil-us.xlsx (US oil prices with 'Monthly' sheet)
- oil_ml_data.xlsx (ML dataset with multiple economic variables)

Output Files:
- Multiple CSV files with prediction results and model metrics
- Feature importance analysis
- Model performance statistics
- Prediction accuracy reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_datasets():
    """Load and prepare the three datasets for XGBoost modeling"""
    print("="*60)
    print("XGBOOST MODEL FOR OIL PRICE PREDICTION")
    print("="*60)

    print("\nPhase 1: Data Loading and Preparation")
    print("-" * 40)

    # Load EU oil prices dataset
    oil_eu = pd.read_excel('crude-oil-eu.xlsx', sheet_name='Monthly')
    print(f"EU Oil Dataset loaded: {oil_eu.shape}")

    # Load US oil prices dataset  
    oil_us = pd.read_excel('crude-oil-us.xlsx', sheet_name='Monthly')
    print(f"US Oil Dataset loaded: {oil_us.shape}")

    # Load ML dataset with economic indicators
    oil_ml = pd.read_excel('oil_ml_data.xlsx', sheet_name='data')
    print(f"ML Dataset loaded: {oil_ml.shape}")

    # Convert date columns to datetime
    oil_eu['observation_date'] = pd.to_datetime(oil_eu['observation_date'])
    oil_us['observation_date'] = pd.to_datetime(oil_us['observation_date'])
    oil_ml['Date'] = pd.to_datetime(oil_ml['Date'])

    # Rename columns for consistency
    oil_eu = oil_eu.rename(columns={'observation_date': 'Date', 'MCOILBRENTEU': 'EU_Oil_Price'})
    oil_us = oil_us.rename(columns={'observation_date': 'Date', 'DCOILWTICO': 'US_Oil_Price'})

    # Remove missing values from price datasets
    oil_us = oil_us.dropna()
    oil_eu = oil_eu.dropna()

    print(f"After cleaning - US Oil: {oil_us.shape}, EU Oil: {oil_eu.shape}")

    # Find common date range
    common_start = max(oil_eu['Date'].min(), oil_us['Date'].min(), oil_ml['Date'].min())
    common_end = min(oil_eu['Date'].max(), oil_us['Date'].max(), oil_ml['Date'].max())

    print(f"Common date range: {common_start.strftime('%Y-%m')} to {common_end.strftime('%Y-%m')}")

    # Filter datasets to common date range
    oil_eu_filtered = oil_eu[(oil_eu['Date'] >= common_start) & (oil_eu['Date'] <= common_end)]
    oil_us_filtered = oil_us[(oil_us['Date'] >= common_start) & (oil_us['Date'] <= common_end)]
    oil_ml_filtered = oil_ml[(oil_ml['Date'] >= common_start) & (oil_ml['Date'] <= common_end)]

    return oil_eu_filtered, oil_us_filtered, oil_ml_filtered

def create_target_variables(oil_us_filtered, oil_eu_filtered):
    """Create target variables for prediction"""
    print("\nCreating target variables...")

    # Merge US and EU oil prices
    oil_prices = pd.merge(oil_us_filtered[['Date', 'US_Oil_Price']], 
                          oil_eu_filtered[['Date', 'EU_Oil_Price']], 
                          on='Date', how='inner')

    # Create multiple target variables
    oil_prices['Global_Oil_Price'] = (oil_prices['US_Oil_Price'] + oil_prices['EU_Oil_Price']) / 2
    oil_prices['Price_Volatility'] = oil_prices['Global_Oil_Price'].rolling(window=3).std()
    oil_prices['Price_Return'] = oil_prices['Global_Oil_Price'].pct_change()
    oil_prices['Price_Direction'] = (oil_prices['Global_Oil_Price'].diff() > 0).astype(int)

    # Create lag features for target
    oil_prices['Global_Oil_Price_lag1'] = oil_prices['Global_Oil_Price'].shift(1)
    oil_prices['Global_Oil_Price_lag2'] = oil_prices['Global_Oil_Price'].shift(2)
    oil_prices['Global_Oil_Price_lag3'] = oil_prices['Global_Oil_Price'].shift(3)

    # Moving averages
    oil_prices['Price_MA_3'] = oil_prices['Global_Oil_Price'].rolling(window=3).mean()
    oil_prices['Price_MA_6'] = oil_prices['Global_Oil_Price'].rolling(window=6).mean()
    oil_prices['Price_MA_12'] = oil_prices['Global_Oil_Price'].rolling(window=12).mean()

    print(f"Oil prices with targets shape: {oil_prices.shape}")
    print("Target variables created:")
    print("• Global_Oil_Price - Main prediction target")
    print("• Price_Volatility - 3-month rolling volatility")
    print("• Price_Return - Monthly return")
    print("• Price_Direction - Binary direction (up/down)")

    return oil_prices

def engineer_features(oil_ml_filtered, oil_prices):
    """Comprehensive feature engineering"""
    print("\nPhase 2: Feature Engineering")
    print("-" * 40)

    # Merge datasets
    ml_data = pd.merge(oil_prices, oil_ml_filtered, on='Date', how='inner')

    print(f"Combined dataset shape: {ml_data.shape}")

    # Handle missing values in China PMI
    ml_data['china_pmi'] = ml_data['china_pmi'].fillna(method='bfill').fillna(method='ffill')

    # Feature engineering for economic indicators
    feature_cols = ['global_pmi', 'oecd_oil_inventories', 'opec_production', 'us_crude_production',
                   'usd_index', 'fed_funds_rate', 'china_pmi', 'eu_pmi', 'india_iip', 
                   'cpi_global', 'rig_count_us', 'refinery_utilization_us']

    print(f"Base economic features: {len(feature_cols)}")

    # Create lag features for key economic indicators
    lag_features = ['global_pmi', 'opec_production', 'us_crude_production', 'usd_index', 
                   'fed_funds_rate', 'china_pmi', 'eu_pmi']

    for feature in lag_features:
        if feature in ml_data.columns:
            ml_data[f'{feature}_lag1'] = ml_data[feature].shift(1)
            ml_data[f'{feature}_lag2'] = ml_data[feature].shift(2)

    # Create change/difference features
    change_features = ['global_pmi', 'oecd_oil_inventories', 'opec_production', 'usd_index', 'cpi_global']

    for feature in change_features:
        if feature in ml_data.columns:
            ml_data[f'{feature}_change'] = ml_data[feature].diff()
            ml_data[f'{feature}_pct_change'] = ml_data[feature].pct_change()

    # Create moving averages for key indicators
    ma_features = ['global_pmi', 'opec_production', 'us_crude_production', 'fed_funds_rate']

    for feature in ma_features:
        if feature in ml_data.columns:
            ml_data[f'{feature}_ma3'] = ml_data[feature].rolling(window=3).mean()
            ml_data[f'{feature}_ma6'] = ml_data[feature].rolling(window=6).mean()

    # Create interaction features
    if 'global_pmi' in ml_data.columns and 'china_pmi' in ml_data.columns:
        ml_data['pmi_interaction'] = ml_data['global_pmi'] * ml_data['china_pmi']

    if 'opec_production' in ml_data.columns and 'us_crude_production' in ml_data.columns:
        ml_data['total_production'] = ml_data['opec_production'] + ml_data['us_crude_production']
        ml_data['production_ratio'] = ml_data['opec_production'] / (ml_data['us_crude_production'] + 1e-6)

    # Create time-based features
    ml_data['Year'] = ml_data['Date'].dt.year
    ml_data['Month'] = ml_data['Date'].dt.month
    ml_data['Quarter'] = ml_data['Date'].dt.quarter
    ml_data['Year_Month'] = ml_data['Year'] * 100 + ml_data['Month']

    # Create cyclical features
    ml_data['Month_sin'] = np.sin(2 * np.pi * ml_data['Month'] / 12)
    ml_data['Month_cos'] = np.cos(2 * np.pi * ml_data['Month'] / 12)
    ml_data['Quarter_sin'] = np.sin(2 * np.pi * ml_data['Quarter'] / 4)
    ml_data['Quarter_cos'] = np.cos(2 * np.pi * ml_data['Quarter'] / 4)

    print(f"Final dataset shape after feature engineering: {ml_data.shape}")

    # Remove rows with too many missing values (from lag features)
    ml_data = ml_data.dropna()
    print(f"After removing NaN: {ml_data.shape}")

    return ml_data

def prepare_modeling_data(ml_data, target_col='Global_Oil_Price'):
    """Prepare data for XGBoost modeling"""
    print("\nPhase 3: Preparing Data for Modeling")
    print("-" * 40)

    # Define features to exclude (non-predictive or target-related)
    exclude_cols = ['Date', 'US_Oil_Price', 'EU_Oil_Price', 'Global_Oil_Price', 
                   'Price_Volatility', 'Price_Return', 'Price_Direction', 'target_oil_price',
                   'Year_Month']  # Year_Month might cause overfitting

    # Get feature columns
    feature_cols = [col for col in ml_data.columns if col not in exclude_cols]

    print(f"Total features available: {len(feature_cols)}")

    # Prepare X and y
    X = ml_data[feature_cols].copy()
    y = ml_data[target_col].copy()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # Check for any remaining missing values
    missing_summary = X.isnull().sum()
    if missing_summary.sum() > 0:
        print("\nMissing values found in features:")
        print(missing_summary[missing_summary > 0])

        # Fill remaining missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        print("Missing values filled using forward/backward fill")

    # Feature selection using statistical tests
    print("\nPerforming feature selection...")
    selector = SelectKBest(f_regression, k=min(50, X.shape[1]))  # Select top 50 features
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()

    print(f"Selected {len(selected_features)} most important features")

    # Create DataFrame with selected features
    X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    return X_final, y, selected_features, ml_data['Date']

def split_data_time_series(X, y, dates, test_size=0.2):
    """Split data maintaining time series order"""
    print("\nPhase 4: Time Series Data Splitting")
    print("-" * 40)

    # Sort by date to maintain temporal order
    sort_idx = dates.argsort()
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)
    dates_sorted = dates.iloc[sort_idx].reset_index(drop=True)

    # Time series split - use last portion for testing
    split_point = int(len(X_sorted) * (1 - test_size))

    X_train = X_sorted.iloc[:split_point]
    X_test = X_sorted.iloc[split_point:]
    y_train = y_sorted.iloc[:split_point]
    y_test = y_sorted.iloc[split_point:]
    dates_train = dates_sorted.iloc[:split_point]
    dates_test = dates_sorted.iloc[split_point:]

    print(f"Training set: {X_train.shape[0]} samples ({dates_train.min().strftime('%Y-%m')} to {dates_train.max().strftime('%Y-%m')})")
    print(f"Test set: {X_test.shape[0]} samples ({dates_test.min().strftime('%Y-%m')} to {dates_test.max().strftime('%Y-%m')})")

    return X_train, X_test, y_train, y_test, dates_train, dates_test

def scale_features(X_train, X_test):
    """Scale features for better model performance"""
    print("\nScaling features...")

    # Use RobustScaler as it's less sensitive to outliers
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )

    print("Features scaled using RobustScaler")

    return X_train_scaled, X_test_scaled, scaler

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using TimeSeriesSplit"""
    print("\nPhase 5: Hyperparameter Tuning")
    print("-" * 40)

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    print(f"Parameter grid size: {len(param_grid)} parameters")

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

    print("Performing grid search with time series cross-validation...")
    print("This may take several minutes...")

    # Perform grid search (using a reduced grid for efficiency)
    reduced_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        xgb_model, 
        reduced_param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"Best CV score (RMSE): {np.sqrt(-grid_search.best_score_):.4f}")

    return grid_search.best_estimator_, grid_search.best_params_

def train_final_model(X_train, y_train, best_params):
    """Train the final XGBoost model"""
    print("\nPhase 6: Training Final XGBoost Model")
    print("-" * 40)

    # Create model with best parameters
    final_model = xgb.XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        **best_params
    )

    # Train the model
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )

    print("Final XGBoost model trained successfully")
    print(f"Number of trees: {final_model.n_estimators}")
    print(f"Max depth: {final_model.max_depth}")
    print(f"Learning rate: {final_model.learning_rate}")

    return final_model

def evaluate_model(model, X_train, X_test, y_train, y_test, dates_train, dates_test):
    """Comprehensive model evaluation"""
    print("\nPhase 7: Model Evaluation")
    print("-" * 40)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics for training set
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate metrics for test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Model Performance Metrics:")
    print("="*30)
    print(f"Training Set:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")

    print(f"\nTest Set:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")

    # Calculate additional metrics
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    print(f"  MAPE: {mape_test:.2f}%")

    # Directional accuracy (for price movement)
    y_test_direction = (y_test.diff() > 0).iloc[1:]
    y_pred_direction = (pd.Series(y_test_pred, index=y_test.index).diff() > 0).iloc[1:]
    directional_accuracy = (y_test_direction == y_pred_direction).mean() * 100
    print(f"  Directional Accuracy: {directional_accuracy:.1f}%")

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test.values,
        'Predicted': y_test_pred,
        'Error': y_test.values - y_test_pred,
        'Abs_Error': np.abs(y_test.values - y_test_pred),
        'Pct_Error': ((y_test.values - y_test_pred) / y_test.values) * 100
    })

    # Model evaluation metrics
    evaluation_metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_mape': mape_test,
        'directional_accuracy': directional_accuracy
    }

    return results_df, evaluation_metrics, y_train_pred, y_test_pred

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    print("\nPhase 8: Feature Importance Analysis")
    print("-" * 40)

    # Get feature importance
    importance_scores = model.feature_importances_

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)

    print("Top 15 Most Important Features:")
    print("-" * 35)
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:<30} {row['Importance']:.4f}")

    # Categorize features
    feature_categories = {
        'PMI': ['pmi', 'PMI'],
        'Production': ['production', 'rig_count'],
        'Financial': ['usd_index', 'fed_funds_rate'],
        'Inflation': ['cpi'],
        'Time': ['Month', 'Quarter', 'Year'],
        'Lags': ['lag'],
        'Changes': ['change', 'pct_change'],
        'Moving_Avg': ['ma3', 'ma6'],
        'Oil_History': ['Oil_Price_lag'],
        'Other': []
    }

    # Categorize features
    importance_df['Category'] = 'Other'
    for category, keywords in feature_categories.items():
        for keyword in keywords:
            mask = importance_df['Feature'].str.contains(keyword, case=False)
            importance_df.loc[mask, 'Category'] = category

    # Category-wise importance
    category_importance = importance_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)

    print("\nImportance by Feature Category:")
    print("-" * 32)
    for category, importance in category_importance.items():
        print(f"{category:<15} {importance:.4f}")

    return importance_df, category_importance

def create_predictions_timeline(results_df, dates_train, y_train, y_train_pred):
    """Create timeline of predictions vs actuals"""
    print("\nCreating prediction timeline...")

    # Combine training and test results
    train_results = pd.DataFrame({
        'Date': dates_train,
        'Actual': y_train.values,
        'Predicted': y_train_pred,
        'Set': 'Train'
    })

    test_results = results_df[['Date', 'Actual', 'Predicted']].copy()
    test_results['Set'] = 'Test'

    full_timeline = pd.concat([train_results, test_results], ignore_index=True)
    full_timeline = full_timeline.sort_values('Date').reset_index(drop=True)

    return full_timeline

def save_results(results_df, evaluation_metrics, importance_df, category_importance, 
                full_timeline, model, best_params, selected_features):
    """Save all results to CSV files"""
    print("\nPhase 9: Saving Results")
    print("-" * 40)

    try:
        # 1. Prediction results
        results_df.to_csv('xgboost_predictions.csv', index=False)
        print("✓ Prediction results saved to xgboost_predictions.csv")

        # 2. Model metrics
        metrics_df = pd.DataFrame([evaluation_metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_csv('xgboost_metrics.csv')
        print("✓ Model metrics saved to xgboost_metrics.csv")

        # 3. Feature importance
        importance_df.to_csv('xgboost_feature_importance.csv', index=False)
        print("✓ Feature importance saved to xgboost_feature_importance.csv")

        # 4. Category importance
        category_df = pd.DataFrame({'Category': category_importance.index, 
                                  'Importance': category_importance.values})
        category_df.to_csv('xgboost_category_importance.csv', index=False)
        print("✓ Category importance saved to xgboost_category_importance.csv")

        # 5. Full timeline
        full_timeline.to_csv('xgboost_full_timeline.csv', index=False)
        print("✓ Full prediction timeline saved to xgboost_full_timeline.csv")

        # 6. Model configuration
        config_df = pd.DataFrame([best_params]).T
        config_df.columns = ['Value']
        config_df.to_csv('xgboost_model_config.csv')
        print("✓ Model configuration saved to xgboost_model_config.csv")

        # 7. Selected features list
        features_df = pd.DataFrame({'Selected_Features': selected_features})
        features_df.to_csv('xgboost_selected_features.csv', index=False)
        print("✓ Selected features saved to xgboost_selected_features.csv")

        print(f"\nAll XGBoost results saved successfully!")

    except Exception as e:
        print(f"Error saving results: {e}")

def print_final_summary(evaluation_metrics, importance_df, category_importance, best_params):
    """Print comprehensive final summary"""
    print("\n" + "="*60)
    print("XGBOOST MODEL FINAL SUMMARY")
    print("="*60)

    print(f"\n1. MODEL PERFORMANCE:")
    print(f"   • Test RMSE: {evaluation_metrics['test_rmse']:.4f}")
    print(f"   • Test MAE: {evaluation_metrics['test_mae']:.4f}")  
    print(f"   • Test R²: {evaluation_metrics['test_r2']:.4f}")
    print(f"   • Test MAPE: {evaluation_metrics['test_mape']:.2f}%")
    print(f"   • Directional Accuracy: {evaluation_metrics['directional_accuracy']:.1f}%")

    print(f"\n2. MODEL CONFIGURATION:")
    print(f"   • Algorithm: XGBoost Regressor")
    for param, value in best_params.items():
        print(f"   • {param}: {value}")

    print(f"\n3. TOP 5 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"   {i}. {row['Feature']} ({row['Importance']:.4f})")

    print(f"\n4. FEATURE CATEGORIES BY IMPORTANCE:")
    for i, (category, importance) in enumerate(category_importance.head(5).items(), 1):
        print(f"   {i}. {category}: {importance:.4f}")

    print(f"\n5. MODEL INSIGHTS:")
    if evaluation_metrics['test_r2'] > 0.7:
        print(f"   • Strong predictive performance (R² > 0.7)")
    elif evaluation_metrics['test_r2'] > 0.5:
        print(f"   • Good predictive performance (R² > 0.5)")
    else:
        print(f"   • Moderate predictive performance")

    if evaluation_metrics['directional_accuracy'] > 60:
        print(f"   • Good directional prediction capability")
    else:
        print(f"   • Limited directional prediction capability")

    print(f"   • Model complexity: {best_params.get('n_estimators', 'N/A')} trees")

    print(f"\n6. OUTPUT FILES GENERATED:")
    print(f"   • xgboost_predictions.csv - Test set predictions")
    print(f"   • xgboost_metrics.csv - Model performance metrics")
    print(f"   • xgboost_feature_importance.csv - Feature importance scores")
    print(f"   • xgboost_category_importance.csv - Category-wise importance")
    print(f"   • xgboost_full_timeline.csv - Complete prediction timeline")
    print(f"   • xgboost_model_config.csv - Model hyperparameters")
    print(f"   • xgboost_selected_features.csv - List of selected features")

def main():
    """Main execution function"""
    print("XGBoost Oil Price Prediction Model")
    print("Starting comprehensive ML analysis...")

    try:
        # Phase 1: Load and prepare datasets
        oil_eu_filtered, oil_us_filtered, oil_ml_filtered = load_and_prepare_datasets()

        # Phase 2: Create target variables
        oil_prices = create_target_variables(oil_us_filtered, oil_eu_filtered)

        # Phase 3: Feature engineering
        ml_data = engineer_features(oil_ml_filtered, oil_prices)

        # Phase 4: Prepare modeling data
        X, y, selected_features, dates = prepare_modeling_data(ml_data)

        # Phase 5: Split data maintaining time series order
        X_train, X_test, y_train, y_test, dates_train, dates_test = split_data_time_series(X, y, dates)

        # Phase 6: Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # Phase 7: Hyperparameter tuning
        best_model, best_params = hyperparameter_tuning(X_train_scaled, y_train)

        # Phase 8: Train final model
        final_model = train_final_model(X_train_scaled, y_train, best_params)

        # Phase 9: Evaluate model
        results_df, evaluation_metrics, y_train_pred, y_test_pred = evaluate_model(
            final_model, X_train_scaled, X_test_scaled, y_train, y_test, dates_train, dates_test)

        # Phase 10: Feature importance analysis
        importance_df, category_importance = analyze_feature_importance(final_model, selected_features)

        # Phase 11: Create full timeline
        full_timeline = create_predictions_timeline(results_df, dates_train, y_train, y_train_pred)

        # Phase 12: Save results
        save_results(results_df, evaluation_metrics, importance_df, category_importance,
                    full_timeline, final_model, best_params, selected_features)

        # Phase 13: Final summary
        print_final_summary(evaluation_metrics, importance_df, category_importance, best_params)

        print("\n" + "="*60)
        print("XGBOOST ANALYSIS COMPLETE!")
        print("="*60)

        return final_model, results_df, evaluation_metrics

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, results, metrics = main()
