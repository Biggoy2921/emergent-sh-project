#!/usr/bin/env python3
"""
Generate synthetic malware dataset based on CIC methodology
Dataset specs: 17,394 samples, 279 features (reduced to 50 for efficiency)
Malware families: Backdoor, Rootkit, Virus, Trojan, Exploit, Clean
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def generate_pe_features(n_samples, malware_type):
    """Generate synthetic PE header features"""
    np.random.seed(42 if malware_type == 'Clean' else hash(malware_type) % 10000)
    
    # Define feature characteristics based on malware type
    if malware_type == 'Clean':
        # Clean files have lower entropy, normal section counts
        entropy = np.random.uniform(3.0, 5.5, n_samples)
        num_sections = np.random.randint(2, 6, n_samples)
        num_imports = np.random.randint(50, 200, n_samples)
        suspicious_imports = np.random.randint(0, 3, n_samples)
        code_size = np.random.uniform(10000, 500000, n_samples)
        
    elif malware_type == 'Backdoor':
        entropy = np.random.uniform(6.5, 7.8, n_samples)
        num_sections = np.random.randint(4, 10, n_samples)
        num_imports = np.random.randint(150, 400, n_samples)
        suspicious_imports = np.random.randint(5, 20, n_samples)
        code_size = np.random.uniform(50000, 1000000, n_samples)
        
    elif malware_type == 'Rootkit':
        entropy = np.random.uniform(7.0, 7.9, n_samples)
        num_sections = np.random.randint(5, 12, n_samples)
        num_imports = np.random.randint(200, 500, n_samples)
        suspicious_imports = np.random.randint(10, 30, n_samples)
        code_size = np.random.uniform(100000, 2000000, n_samples)
        
    elif malware_type == 'Virus':
        entropy = np.random.uniform(6.0, 7.5, n_samples)
        num_sections = np.random.randint(3, 8, n_samples)
        num_imports = np.random.randint(100, 350, n_samples)
        suspicious_imports = np.random.randint(8, 25, n_samples)
        code_size = np.random.uniform(20000, 800000, n_samples)
        
    elif malware_type == 'Trojan':
        entropy = np.random.uniform(6.8, 7.9, n_samples)
        num_sections = np.random.randint(4, 11, n_samples)
        num_imports = np.random.randint(180, 450, n_samples)
        suspicious_imports = np.random.randint(12, 35, n_samples)
        code_size = np.random.uniform(80000, 1500000, n_samples)
        
    else:  # Exploit
        entropy = np.random.uniform(7.2, 7.99, n_samples)
        num_sections = np.random.randint(6, 15, n_samples)
        num_imports = np.random.randint(250, 600, n_samples)
        suspicious_imports = np.random.randint(15, 40, n_samples)
        code_size = np.random.uniform(150000, 3000000, n_samples)
    
    # Generate 50 features (reduced from 279 for efficiency)
    features = {
        'entropy': entropy,
        'num_sections': num_sections,
        'num_imports': num_imports,
        'suspicious_imports': suspicious_imports,
        'code_size': code_size,
        'data_size': np.random.uniform(5000, 200000, n_samples),
        'bss_size': np.random.uniform(0, 50000, n_samples),
        'virtual_size': code_size * np.random.uniform(1.1, 2.5, n_samples),
        'num_exported_functions': np.random.randint(0, 50, n_samples),
        'has_tls': np.random.randint(0, 2, n_samples),
        'has_resources': np.random.randint(0, 2, n_samples),
        'num_resources': np.random.randint(0, 100, n_samples),
        'image_base': np.random.choice([0x00400000, 0x10000000, 0x00140000], n_samples),
        'section_entropy_mean': entropy + np.random.uniform(-0.5, 0.5, n_samples),
        'section_entropy_std': np.random.uniform(0.1, 1.5, n_samples),
    }
    
    # Add 35 more statistical features
    for i in range(35):
        if malware_type == 'Clean':
            features[f'feature_{i}'] = np.random.uniform(0, 0.4, n_samples)
        else:
            features[f'feature_{i}'] = np.random.uniform(0.3, 1.0, n_samples)
    
    df = pd.DataFrame(features)
    df['malware_type'] = malware_type
    df['is_malware'] = 0 if malware_type == 'Clean' else 1
    
    return df

def generate_dataset():
    """Generate complete dataset with all malware families"""
    print("🔬 Generating synthetic malware dataset...")
    
    # Distribution based on paper (17,394 samples)
    samples_per_type = {
        'Clean': 5000,
        'Backdoor': 2500,
        'Rootkit': 2000,
        'Virus': 2894,
        'Trojan': 3000,
        'Exploit': 2000
    }
    
    dfs = []
    for malware_type, n_samples in samples_per_type.items():
        print(f"  Generating {n_samples} {malware_type} samples...")
        df = generate_pe_features(n_samples, malware_type)
        dfs.append(df)
    
    # Combine all data
    full_dataset = pd.concat(dfs, ignore_index=True)
    
    # Shuffle dataset
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Dataset generated: {len(full_dataset)} samples with {len(full_dataset.columns)-2} features")
    print(f"\nMalware distribution:")
    print(full_dataset['malware_type'].value_counts())
    
    return full_dataset

if __name__ == '__main__':
    dataset = generate_dataset()
    
    # Save dataset
    output_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    dataset.to_csv(output_path, index=False)
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Generate train/test split
    X = dataset.drop(['malware_type', 'is_malware'], axis=1)
    y_binary = dataset['is_malware']
    y_multiclass = dataset['malware_type']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
