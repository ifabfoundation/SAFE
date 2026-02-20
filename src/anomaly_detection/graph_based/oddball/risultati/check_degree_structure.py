#!/usr/bin/env python
import pickle
from pathlib import Path

degree_file = Path('/home/projects/safe/outputs/networks/degree/degree_distributions.pkl')

with open(degree_file, 'rb') as f:
    degree_data = pickle.load(f)

print("Type of degree_data:", type(degree_data))
print("\nKeys:", list(degree_data.keys())[:3])

# Check structure of one entry
key = 'G_NVG_p3_temp'
entry = degree_data[key]
print(f"\nType of degree_data['{key}']:", type(entry))

if isinstance(entry, dict):
    sample_items = list(entry.items())[:5]
    print(f"First 5 items: {sample_items}")
elif isinstance(entry, (list, tuple)):
    print(f"Length: {len(entry)}")
    print(f"First 5 items: {entry[:5]}")
else:
    print(f"Content: {entry}")
