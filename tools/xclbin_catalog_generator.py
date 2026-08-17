#!/usr/bin/env python3
"""Systematic analysis of all NPU2 xclbin kernels."""
import sys, os, json from pathlib import Path

sys.path.insert(0, str(Path.cwd() + "/tools"))
