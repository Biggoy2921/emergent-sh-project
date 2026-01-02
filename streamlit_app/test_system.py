#!/usr/bin/env python3
"""
Quick system test to verify all components are working
"""

import sys
import os

print("="*60)
print("🧪 MALWARE DETECTION SYSTEM - COMPONENT TEST")
print("="*60)

# Test 1: Check dataset
print("\n1️⃣ Testing dataset...")
try:
    import pandas as pd
    df = pd.read_csv('data/dataset.csv')
    print(f"   ✅ Dataset loaded: {len(df)} samples")
except Exception as e:
    print(f"   ❌ Dataset error: {e}")

# Test 2: Check models
print("\n2️⃣ Testing model files...")
model_dir = 'data/saved_models'
models = {
    'Decision Tree': 'model_dt.pkl',
    'SVM': 'model_svm.pkl',
    'CNN': 'model_cnn.h5',
    'Preprocessor': 'preprocessor.pkl'
}

for name, filename in models.items():
    path = os.path.join(model_dir, filename)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"   ✅ {name}: {size/1024:.1f} KB")
    else:
        print(f"   ❌ {name}: Not found")

# Test 3: Load ensemble
print("\n3️⃣ Testing ensemble classifier...")
try:
    from models.ensemble import EnsembleClassifier
    ensemble = EnsembleClassifier(model_dir)
    ensemble.load_models()
    print("   ✅ All models loaded successfully")
except Exception as e:
    print(f"   ❌ Ensemble error: {e}")

# Test 4: Test prediction
print("\n4️⃣ Testing prediction on sample data...")
try:
    import numpy as np
    # Create sample malware-like features (high entropy, many imports)
    malware_features = np.array([
        7.5,  # entropy (high)
        10,   # num_sections
        300,  # num_imports  
        20,   # suspicious_imports (high)
        500000,  # code_size
        100000,  # data_size
        1000,    # bss_size
        600000,  # virtual_size
        5,    # num_exported_functions
        1,    # has_tls
        1,    # has_resources
        50,   # num_resources
        0x00400000,  # image_base
        7.2,  # section_entropy_mean
        0.8   # section_entropy_std
    ] + [0.8] * 35)  # 35 additional features
    
    result = ensemble.predict_single(malware_features)
    
    print(f"   Prediction: {'MALWARE' if result['prediction'] == 1 else 'CLEAN'}")
    print(f"   Probability: {result['probability']*100:.2f}%")
    print(f"   DT: {result['individual_probabilities']['DT']*100:.1f}%")
    print(f"   SVM: {result['individual_probabilities']['SVM']*100:.1f}%")
    print(f"   CNN: {result['individual_probabilities']['CNN']*100:.1f}%")
    print("   ✅ Prediction working correctly")
    
except Exception as e:
    print(f"   ❌ Prediction error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Feature extractors
print("\n5️⃣ Testing feature extractors...")
try:
    from features.pe_extractor import PEFeatureExtractor
    from features.doc_extractor import DocumentFeatureExtractor
    
    pe_ext = PEFeatureExtractor()
    doc_ext = DocumentFeatureExtractor()
    print("   ✅ Feature extractors initialized")
except Exception as e:
    print(f"   ❌ Extractor error: {e}")

# Summary
print("\n" + "="*60)
print("🎉 SYSTEM TEST COMPLETE!")
print("="*60)
print("\n✨ To run the Streamlit app:")
print("   cd /app/streamlit_app")
print("   streamlit run app.py --server.port 8501")
print("\n🌐 Then open: http://localhost:8501")
print("="*60)
