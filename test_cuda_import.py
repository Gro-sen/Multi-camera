#!/usr/bin/env python
"""测试 app 模块初始化后 ONNX Runtime CUDA 支持"""

print("[1] 导入 app 模块（应该自动注入 CUDA DLL 目录）")
import app

print("[2] 导入 ONNX Runtime")
import onnxruntime as ort

providers = ort.get_available_providers()
print(f"[3] 可用提供者: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✓ SUCCESS: CUDA 提供者已启用！")
    exit(0)
else:
    print("✗ FAILED: CUDA 提供者未启用")
    exit(1)
