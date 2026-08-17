#!/usr/bin/env python3
"""Systematic analysis of all NPU2 xclbin kernels."""

import os, sys, json, struct
from pathlib import Path

# Add tools to path  
sys.path.insert(0, str(Path(__file__).parent / "tools"))

def analyze_single_xclbin(path):
    """Analyze a single xclbin file."""
    result = {
        "path": str(path),
        "filename": os.path.basename(path),
        "size_bytes": path.stat().st_size,
    }
    
    try:
        from axlf import load as load_axlf
        
        xc = load_axlf(path)
        
        # Basic info  
        result["uuid"] = xc.uuid.hex() if hasattr(xc.uuid, 'hex') else str(xc.uuid)
        result["kernel_name"] = getattr(xc, "kernel_name", "Unknown")
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# Main analysis loop  
xclbin_dir = Path("../FastFlowLM/src/xclbins")
results = []

print("Scanning xclbins...")
for model_dir in xclbin_dir.iterdir():
    if not model_dir.is_dir():
        continue
    
    for ext_file in model_dir.glob("*.xclbin"):
        print(f"\n{'='*60}")
        print(f"File: {ext_file.relative_to(Path.cwd())}")

print("\n\nAnalysis complete! Review individual xclbins for detailed CDO/DMA info.")
