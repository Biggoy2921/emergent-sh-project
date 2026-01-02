#!/usr/bin/env python3
"""
Data Preprocessor
Handles feature scaling and preprocessing for ML models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

class FeaturePreprocessor:
    """Preprocess and scale features for malware detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
    
    def fit(self, X, y=None):
        """Fit the preprocessor on training data"""
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        """Transform features"""
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath):
        """Save preprocessor to disk"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load preprocessor from disk"""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_selector = data.get('feature_selector')
        self.selected_features = data.get('selected_features')
        print(f"Preprocessor loaded from {filepath}")
        return self

if __name__ == '__main__':
    # Test preprocessor
    print("🔧 Testing Feature Preprocessor...")
    
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        X = df.drop(['malware_type', 'is_malware'], axis=1)
        y = df['is_malware']
        
        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(X, y)
        
        print(f"Original shape: {X.shape}")
        print(f"Scaled shape: {X_scaled.shape}")
        print(f"Scaled mean: {X_scaled.mean():.4f}")
        print(f"Scaled std: {X_scaled.std():.4f}")
        
        # Save preprocessor
        save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models', 'preprocessor.pkl')
        preprocessor.save(save_path)
    else:
        print(f"Dataset not found at {data_path}")
        print("Please run generate_dataset.py first")
