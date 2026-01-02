#!/usr/bin/env python3
"""
PE (Portable Executable) Feature Extractor
Extracts static analysis features from executable files (.exe, .dll)
"""

import pefile
import math
import os
import numpy as np
from collections import Counter

class PEFeatureExtractor:
    """Extract features from PE files for malware detection"""
    
    def __init__(self):
        self.feature_names = [
            'entropy', 'num_sections', 'num_imports', 'suspicious_imports',
            'code_size', 'data_size', 'bss_size', 'virtual_size',
            'num_exported_functions', 'has_tls', 'has_resources', 'num_resources',
            'image_base', 'section_entropy_mean', 'section_entropy_std'
        ]
        
        # Suspicious API imports commonly used by malware
        self.suspicious_apis = [
            'CreateRemoteThread', 'VirtualAllocEx', 'WriteProcessMemory',
            'SetWindowsHookEx', 'GetProcAddress', 'LoadLibrary',
            'WinExec', 'ShellExecute', 'URLDownloadToFile',
            'InternetOpen', 'InternetReadFile', 'CreateProcess',
            'RegSetValue', 'RegCreateKey', 'CryptEncrypt'
        ]
    
    def calculate_entropy(self, data):
        """Calculate Shannon entropy of byte data"""
        if not data:
            return 0
        
        entropy = 0
        counter = Counter(data)
        length = len(data)
        
        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def extract_features(self, file_path):
        """Extract all PE features from executable file"""
        try:
            pe = pefile.PE(file_path)
            
            features = {}
            
            # 1. File entropy
            with open(file_path, 'rb') as f:
                file_data = f.read()
            features['entropy'] = self.calculate_entropy(file_data)
            
            # 2. Number of sections
            features['num_sections'] = len(pe.sections)
            
            # 3. Section sizes
            features['code_size'] = 0
            features['data_size'] = 0
            features['bss_size'] = 0
            features['virtual_size'] = 0
            
            section_entropies = []
            for section in pe.sections:
                section_name = section.Name.decode().rstrip('\x00')
                size = section.SizeOfRawData
                vsize = section.Misc_VirtualSize
                
                features['virtual_size'] += vsize
                
                # Calculate section entropy
                section_data = section.get_data()
                section_entropy = self.calculate_entropy(section_data)
                section_entropies.append(section_entropy)
                
                # Categorize sections
                if '.text' in section_name or 'CODE' in section_name:
                    features['code_size'] += size
                elif '.data' in section_name or 'DATA' in section_name:
                    features['data_size'] += size
                elif '.bss' in section_name:
                    features['bss_size'] += size
            
            # 4. Section entropy statistics
            features['section_entropy_mean'] = np.mean(section_entropies) if section_entropies else 0
            features['section_entropy_std'] = np.std(section_entropies) if section_entropies else 0
            
            # 5. Import analysis
            features['num_imports'] = 0
            features['suspicious_imports'] = 0
            
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    features['num_imports'] += len(entry.imports)
                    
                    # Check for suspicious APIs
                    for imp in entry.imports:
                        if imp.name:
                            import_name = imp.name.decode() if isinstance(imp.name, bytes) else imp.name
                            if any(sus_api in import_name for sus_api in self.suspicious_apis):
                                features['suspicious_imports'] += 1
            
            # 6. Export analysis
            features['num_exported_functions'] = 0
            if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
                features['num_exported_functions'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
            
            # 7. TLS (Thread Local Storage)
            features['has_tls'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0
            
            # 8. Resources
            features['has_resources'] = 0
            features['num_resources'] = 0
            if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
                features['has_resources'] = 1
                features['num_resources'] = len(pe.DIRECTORY_ENTRY_RESOURCE.entries)
            
            # 9. Image base
            features['image_base'] = pe.OPTIONAL_HEADER.ImageBase
            
            # Add 35 placeholder features to match dataset (50 features total)
            for i in range(35):
                # These would be additional PE features in production
                features[f'feature_{i}'] = np.random.uniform(0, 1)
            
            pe.close()
            return features, None
            
        except Exception as e:
            return None, str(e)
    
    def extract_features_array(self, file_path):
        """Extract features and return as numpy array in correct order"""
        features, error = self.extract_features(file_path)
        
        if error:
            return None, error
        
        # Create ordered feature array (50 features)
        feature_array = []
        
        # Add first 15 named features
        for name in self.feature_names:
            feature_array.append(features.get(name, 0))
        
        # Add remaining 35 features
        for i in range(35):
            feature_array.append(features.get(f'feature_{i}', 0))
        
        return np.array(feature_array).reshape(1, -1), None
