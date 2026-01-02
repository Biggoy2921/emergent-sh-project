#!/usr/bin/env python3
"""
Support Vector Machine Model Training
Target: 96.41% accuracy (as per paper)
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.preprocessor import FeaturePreprocessor

def train_svm():
    """Train SVM model"""
    print("\n" + "="*60)
    print("⚡ Training Support Vector Machine Model")
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
    
    # Prepare data
    X = df.drop(['malware_type', 'is_malware'], axis=1)
    y = df['is_malware']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🔄 Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Load preprocessor
    print("\n⚙️ Loading preprocessor...")
    preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models', 'preprocessor.pkl')
    
    if os.path.exists(preprocessor_path):
        preprocessor = FeaturePreprocessor()
        preprocessor.load(preprocessor_path)
    else:
        print("  Preprocessor not found, creating new one...")
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(X_train, y_train)
    
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Train SVM
    print("\n🎯 Training SVM (this may take a few minutes)...")
    
    model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
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
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models', 'model_svm.pkl')
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("\n✅ SVM training complete!")
    print("="*60)
    
    return model, test_acc

if __name__ == '__main__':
    train_svm()
