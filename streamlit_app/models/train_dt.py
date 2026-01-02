#!/usr/bin/env python3
"""
Decision Tree Model Training
Target: 99% accuracy (as per paper)
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.preprocessor import FeaturePreprocessor

def train_decision_tree():
    """Train Decision Tree model with hyperparameter tuning"""
    print("\n" + "="*60)
    print("🌳 Training Decision Tree Model")
    print("="*60)
    
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
    
    if not os.path.exists(data_path):
        print("\n❌ Dataset not found!")
        print("Please run: python data/generate_dataset.py")
        return
    
    print("\n📊 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(df.columns) - 2}")
    
    # Prepare data
    X = df.drop(['malware_type', 'is_malware'], axis=1)
    y = df['is_malware']
    
    print(f"\n  Clean samples: {(y == 0).sum()}")
    print(f"  Malware samples: {(y == 1).sum()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🔄 Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Preprocess features
    print("\n⚙️ Preprocessing features...")
    preprocessor = FeaturePreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save preprocessor
    preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models', 'preprocessor.pkl')
    preprocessor.save(preprocessor_path)
    
    # Train Decision Tree with optimized hyperparameters
    print("\n🎯 Training Decision Tree...")
    
    # Hyperparameter tuning (commented out for speed, using optimal params)
    # param_grid = {
    #     'max_depth': [10, 20, 30, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'criterion': ['gini', 'entropy']
    # }
    # dt = DecisionTreeClassifier(random_state=42)
    # grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # grid_search.fit(X_train_scaled, y_train)
    # model = grid_search.best_estimator_
    
    # Using optimal hyperparameters for 99% accuracy
    model = DecisionTreeClassifier(
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion='entropy',
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n🏆 RESULTS:")
    print(f"  Training Accuracy: {train_acc*100:.2f}%")
    print(f"  Testing Accuracy: {test_acc*100:.2f}%")
    
    print(f"\n📄 Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Clean', 'Malware']))
    
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models', 'model_dt.pkl')
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models', 'dt_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importances saved to: {importance_path}")
    
    print("\n✅ Decision Tree training complete!")
    print("="*60)
    
    return model, test_acc

if __name__ == '__main__':
    train_decision_tree()
