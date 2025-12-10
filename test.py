#!/usr/bin/env python
"""Simple test script to debug the pipeline"""
import sys
import os

print("=" * 50)
print("TESTING BASIC SETUP")
print("=" * 50)

print(f"\nPython version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

print("\n--- Files in current directory ---")
for item in os.listdir("."):
    print(f"  {item}")

print("\n--- Files in artifacts/ ---")
if os.path.exists("artifacts"):
    for item in os.listdir("artifacts"):
        print(f"  {item}")
else:
    print("  artifacts/ does not exist!")

print("\n--- Checking for data file ---")
paths_to_check = [
    "artifacts/raw_data.csv",
    "notebooks/artifacts/raw_data.csv",
    "notebooks/artifacts/training_data.csv"
]

for path in paths_to_check:
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  ✓ Found: {path} ({size:,} bytes)")
    else:
        print(f"  ✗ Not found: {path}")

print("\n--- Testing imports ---")
try:
    import pandas as pd
    print("  ✓ pandas imported")
except Exception as e:
    print(f"  ✗ pandas failed: {e}")
    sys.exit(1)

try:
    from IPython.display import display
    print("  ✓ IPython.display imported")
except Exception as e:
    print(f"  ✗ IPython.display failed: {e}")
    print("  → This is the likely problem!")

try:
    import sklearn
    print("  ✓ sklearn imported")
except Exception as e:
    print(f"  ✗ sklearn failed: {e}")

print("\n--- Trying to load data ---")
try:
    data = pd.read_csv("artifacts/raw_data.csv")
    print(f"  ✓ Data loaded: {len(data)} rows, {len(data.columns)} columns")
    print(f"  Columns: {list(data.columns)[:5]}...")
except Exception as e:
    print(f"  ✗ Failed to load data: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("ALL CHECKS PASSED!")
print("=" * 50)