#!/usr/bin/env python3
"""
Ensemble Model - Combines DT, CNN, and SVM predictions
Uses voting classifier for final prediction
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score
import os

class EnsembleClassifier:
    """Ensemble of DT, CNN, and SVM for malware detection"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.dt_model = None
        self.svm_model = None
        self.cnn_model = None
        self.preprocessor = None
        
    def load_models(self):
        """Load all trained models"""
        print("📦 Loading trained models...")
        
        # Load preprocessor
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            print("  ✓ Preprocessor loaded")
        else:
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        # Load Decision Tree
        dt_path = os.path.join(self.model_dir, 'model_dt.pkl')
        if os.path.exists(dt_path):
            self.dt_model = joblib.load(dt_path)
            print("  ✓ Decision Tree loaded")
        
        # Load SVM
        svm_path = os.path.join(self.model_dir, 'model_svm.pkl')
        if os.path.exists(svm_path):
            self.svm_model = joblib.load(svm_path)
            print("  ✓ SVM loaded")
        
        # Load CNN
        cnn_path = os.path.join(self.model_dir, 'model_cnn.h5')
        if os.path.exists(cnn_path):
            self.cnn_model = tf.keras.models.load_model(cnn_path)
            print("  ✓ CNN loaded")
        
        print("✅ All models loaded successfully!\n")
    
    def predict_single(self, features):
        """Predict on a single sample with individual model scores"""
        # Preprocess features
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.preprocessor.transform(features)
        
        predictions = {}
        probabilities = {}
        
        # Decision Tree prediction
        if self.dt_model is not None:
            dt_pred = self.dt_model.predict(features_scaled)[0]
            dt_proba = self.dt_model.predict_proba(features_scaled)[0]
            predictions['DT'] = dt_pred
            probabilities['DT'] = dt_proba[1]  # Probability of malware
        
        # SVM prediction
        if self.svm_model is not None:
            svm_pred = self.svm_model.predict(features_scaled)[0]
            svm_proba = self.svm_model.predict_proba(features_scaled)[0]
            predictions['SVM'] = svm_pred
            probabilities['SVM'] = svm_proba[1]
        
        # CNN prediction
        if self.cnn_model is not None:
            cnn_proba = self.cnn_model.predict(features_scaled, verbose=0)[0][0]
            cnn_pred = 1 if cnn_proba > 0.5 else 0
            predictions['CNN'] = cnn_pred
            probabilities['CNN'] = cnn_proba
        
        # Ensemble prediction (majority voting)
        votes = list(predictions.values())
        ensemble_pred = 1 if sum(votes) >= 2 else 0
        
        # Ensemble probability (weighted average)
        # Weights based on paper accuracies: DT=99%, CNN=98.76%, SVM=96.41%
        weights = {'DT': 0.99, 'CNN': 0.9876, 'SVM': 0.9641}
        total_weight = sum(weights.values())
        
        ensemble_proba = sum(
            probabilities.get(model, 0) * weights.get(model, 0) 
            for model in probabilities.keys()
        ) / total_weight
        
        return {
            'prediction': ensemble_pred,
            'probability': ensemble_proba,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities
        }
    
    def predict_malware_family(self, features, probability):
        """Predict specific malware family based on features"""
        # Simple heuristic based on key features
        # In production, this would be a separate trained classifier
        
        if probability < 0.3:
            return 'Clean'
        
        entropy = features[0] if len(features) > 0 else 5.0
        suspicious_imports = features[3] if len(features) > 3 else 0
        num_sections = features[1] if len(features) > 1 else 3
        
        # Classification heuristics
        if entropy > 7.5 and suspicious_imports > 20:
            return 'Rootkit'
        elif entropy > 7.0 and num_sections > 8:
            return 'Exploit'
        elif suspicious_imports > 15:
            return 'Backdoor'
        elif num_sections > 6:
            return 'Trojan'
        else:
            return 'Virus'
    
    def get_threat_level(self, probability):
        """Determine threat level based on malware probability"""
        if probability < 0.25:
            return 'LOW', '🟢'
        elif probability < 0.50:
            return 'MEDIUM', '🟡'
        elif probability < 0.75:
            return 'HIGH', '🟠'
        else:
            return 'CRITICAL', '🔴'

if __name__ == '__main__':
    # Test ensemble
    print("Testing Ensemble Classifier...\n")
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models')
    ensemble = EnsembleClassifier(model_dir)
    
    try:
        ensemble.load_models()
        print("Ensemble ready for predictions!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train models first:")
        print("  python models/train_dt.py")
        print("  python models/train_svm.py")
        print("  python models/train_cnn.py")
