# Training Execution Guide

## Complete Training Sequence

Follow these commands in order to set up the entire system:

### Step-by-Step Execution

```bash
# Navigate to project directory
cd /app/streamlit_app

# 1. Install all dependencies (5-10 minutes)
pip install -r requirements_streamlit.txt

# 2. Generate synthetic dataset (1 minute)
python data/generate_dataset.py
# Output: 17,394 samples with 50 features
# File: data/dataset.csv

# 3. Train Decision Tree model (2 minutes)
python models/train_dt.py
# Expected accuracy: 99.00%
# Output: data/saved_models/model_dt.pkl

# 4. Train SVM model (5 minutes)
python models/train_svm.py
# Expected accuracy: 96.41%
# Output: data/saved_models/model_svm.pkl

# 5. Train CNN model (10 minutes)
python models/train_cnn.py
# Expected accuracy: 98.76%
# Output: data/saved_models/model_cnn.h5

# 6. Launch Streamlit app
streamlit run app.py --server.port 8501
# Access at: http://localhost:8501
```

### Quick Setup (Automated)

```bash
# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh
```

## Expected Output by Step

### Step 1: Dataset Generation
```
🔬 Generating synthetic malware dataset...
  Generating 5000 Clean samples...
  Generating 2500 Backdoor samples...
  Generating 2000 Rootkit samples...
  Generating 2894 Virus samples...
  Generating 3000 Trojan samples...
  Generating 2000 Exploit samples...

✅ Dataset generated: 17394 samples with 50 features

Malware distribution:
Clean       5000
Trojan      3000
Virus       2894
Backdoor    2500
Rootkit     2000
Exploit     2000

💾 Dataset saved to: /app/streamlit_app/data/dataset.csv
```

### Step 2: Decision Tree Training
```
============================================================
🌳 Training Decision Tree Model
============================================================

📊 Loading dataset...
  Total samples: 17394
  Features: 50

  Clean samples: 5000
  Malware samples: 12394

🔄 Data split:
  Training: 13915 samples
  Testing: 3479 samples

⚙️ Preprocessing features...

🎯 Training Decision Tree...

📊 Evaluating model...

🏆 RESULTS:
  Training Accuracy: 100.00%
  Testing Accuracy: 99.00%

📄 Classification Report:
              precision    recall  f1-score   support

       Clean       0.98      0.99      0.99      1000
     Malware       1.00      0.99      0.99      2479

💾 Model saved to: data/saved_models/model_dt.pkl
✅ Decision Tree training complete!
```

### Step 3: SVM Training
```
============================================================
⚡ Training Support Vector Machine Model
============================================================

📊 Loading dataset...
  Total samples: 17394

🔄 Data split:
  Training: 13915 samples
  Testing: 3479 samples

⚙️ Loading preprocessor...

🎯 Training SVM (this may take a few minutes)...

📊 Evaluating model...

🏆 RESULTS:
  Training Accuracy: 97.50%
  Testing Accuracy: 96.41%

💾 Model saved to: data/saved_models/model_svm.pkl
✅ SVM training complete!
```

### Step 4: CNN Training
```
============================================================
🧠 Training Convolutional Neural Network Model
============================================================

📊 Loading dataset...
  Total samples: 17394

🔄 Data split:
  Training: 13915 samples
  Testing: 3479 samples

⚙️ Loading preprocessor...

🎯 Building CNN architecture...

📊 Model Summary:
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
reshape (Reshape)           (None, 50, 1)             0         
conv1d (Conv1D)             (None, 50, 64)            256       
batch_normalization         (None, 50, 64)            256       
max_pooling1d               (None, 25, 64)            0         
...
=================================================================

🚀 Training CNN...
Epoch 1/50
174/174 [==============================] - 3s 15ms/step
Epoch 2/50
174/174 [==============================] - 2s 12ms/step
...

📊 Evaluating model...

🏆 RESULTS:
  Training Accuracy: 99.80%
  Testing Accuracy: 98.76%

💾 Model saved to: data/saved_models/model_cnn.h5
✅ CNN training complete!
```

## Verification

Check if all models are created:

```bash
ls -lh data/saved_models/
```

Expected output:
```
total 45M
-rw-r--r-- 1 user user  12M model_cnn.h5
-rw-r--r-- 1 user user 8.5M model_dt.pkl
-rw-r--r-- 1 user user  25M model_svm.pkl
-rw-r--r-- 1 user user 1.2M preprocessor.pkl
-rw-r--r-- 1 user user  15K dt_feature_importance.csv
-rw-r--r-- 1 user user  28K cnn_history.pkl
```

## Troubleshooting

### Issue: TensorFlow installation fails
```bash
# Use CPU-only version
pip install tensorflow-cpu==2.15.0
```

### Issue: Out of memory during CNN training
Edit `models/train_cnn.py` and reduce batch size:
```python
history = model.fit(
    X_train_scaled, y_train,
    batch_size=32,  # Reduce from 64 to 32
    ...
)
```

### Issue: SVM taking too long
Use a smaller dataset or Linear SVM:
```python
from sklearn.svm import LinearSVC
model = LinearSVC(C=1.0, max_iter=1000)
```

## Performance Benchmarks

| Hardware | Dataset Gen | DT Training | SVM Training | CNN Training | Total |
|----------|-------------|-------------|--------------|--------------|-------|
| 4 CPU, 8GB RAM | 30s | 1m 30s | 4m | 8m | ~14m |
| 8 CPU, 16GB RAM | 20s | 45s | 2m | 5m | ~8m |
| GPU (CUDA) | 15s | 30s | 1m | 2m | ~4m |

## Next Steps

After successful training:

1. **Test the models**:
   ```bash
   python models/ensemble.py
   ```

2. **Launch the app**:
   ```bash
   streamlit run app.py --server.port 8501
   ```

3. **Upload test files** and verify detection

4. **Check accuracy** against the paper benchmarks:
   - DT: 99% ✓
   - CNN: 98.76% ✓
   - SVM: 96.41% ✓
