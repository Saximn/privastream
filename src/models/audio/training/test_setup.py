#!/usr/bin/env python3

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification

print("=== Testing PII Detection Setup ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch version: {torch.__version__}")
print()

# Test data loading
print("Testing data loading...")
try:
    if os.path.exists("data/pii-detection-removal-from-educational-data/train.json"):
        print("[OK] Main competition data found")
        with open("data/pii-detection-removal-from-educational-data/train.json") as f:
            data = json.load(f)
        print(f"[OK] Loaded {len(data)} training samples")
    else:
        print("[ERROR] Main competition data not found")
        
    if os.path.exists("data/pii-dd-mistral-generated/mixtral-8x7b-v1.json"):
        print("[OK] Mixtral data found")
    else:
        print("[ERROR] Mixtral data not found")
        
    if os.path.exists("data/pii-mixtral8x7b-generated-essays/mpware_mixtral8x7b_v1.1-no-i-username.json"):
        print("[OK] MPware data found")
    else:
        print("[ERROR] MPware data not found")
        
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")

print()

# Test model loading
print("Testing model loading...")
try:
    model_name = "microsoft/deberta-v3-large"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("[OK] Tokenizer loaded successfully")
    
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=13)
    model = model.to(device)
    print(f"[OK] Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test a simple forward pass
    print("Testing forward pass...")
    test_text = "Hello, my name is John Smith and my email is john@example.com"
    inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"[OK] Forward pass successful! Output shape: {outputs.logits.shape}")
    
except Exception as e:
    print(f"[ERROR] Error with model: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Setup Test Complete ===")
