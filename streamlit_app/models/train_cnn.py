#!/usr/bin/env python3
"""
Convolutional Neural Network Model Training
Target: 98.76% accuracy (as per paper)
Uses 1D CNN on flattened feature vectors
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.preprocessor import FeaturePreprocessor

def create_cnn_model(input_shape):
    """Create 1D CNN architecture for malware detection"""
    model = models.Sequential([
        # Reshape for 1D CNN
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        # First convolutional block
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Second convolutional block
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_cnn():
    """Train CNN model"""
    print("\n" + "="*60)
    print("🧠 Training Convolutional Neural Network Model")
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
    
    # Create and compile model
    print("\n🎯 Building CNN architecture...")
    input_shape = X_train_scaled.shape[1]
    model = create_cnn_model(input_shape)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📊 Model Summary:")
    model.summary()
    
    # Train model
    print("\n🚀 Training CNN...")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    
    y_train_pred = (model.predict(X_train_scaled) > 0.5).astype(int).flatten()
    y_test_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    
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
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models', 'model_cnn.h5')
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save training history
    history_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_models', 'cnn_history.pkl')
    joblib.dump(history.history, history_path)
    
    print("\n✅ CNN training complete!")
    print("="*60)
    
    return model, test_acc

if __name__ == '__main__':
    train_cnn()
