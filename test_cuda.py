#!/usr/bin/env python
"""快速测试 CUDA 提供者支持"""
import os
import site
from pathlib import Path

# 添加所有 NVIDIA DLL 目录
base_dirs = site.getsitepackages()
for base in base_dirs:
    for nvidia_pkg in ['cublas', 'cufft', 'curand', 'cusolver', 'cusparse', 'cuda_runtime', 'cuda_nvrtc', 'cudnn', 'nvjitlink']:
        dll_bin = Path(base) / 'nvidia' / nvidia_pkg / 'bin'
        if dll_bin.exists():
            try:
                os.add_dll_directory(str(dll_bin))
                print(f"✓ Added: {dll_bin}")
            except Exception as e:
                print(f"✗ Failed to add {dll_bin}: {e}")

print("\n=== ONNX Runtime 提供者检查 ===")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"可用提供者: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("✓ CUDA 提供者已启用")
    else:
        print("✗ CUDA 提供者未启用，使用 CPU")
except Exception as e:
    print(f"✗ 获取提供者失败: {e}")
