#!/usr/bin/env python
"""优化的 GPU 推理测试 - 确保 CUDA DLL 在所有层级可用"""
import os
import sys

# ===== 第 0 步：在导入任何 NVIDIA 库前添加 DLL 路径 =====
if os.name == "nt":
    import site
    from pathlib import Path
    
    print("[0] 添加 NVIDIA DLL 搜索路径到 PATH...")
    site_paths = site.getsitepackages()
    for base in site_paths:
        for rel in (
            "nvidia\\cublas\\bin",
            "nvidia\\cudnn\\bin",
            "nvidia\\cufft\\bin",
            "nvidia\\curand\\bin",
            "nvidia\\cusolver\\bin",
            "nvidia\\cusparse\\bin",
            "nvidia\\cuda_runtime\\bin",
            "nvidia\\cuda_nvrtc\\bin",
            "nvidia\\nvjitlink\\bin",
        ):
            candidate = Path(base) / rel
            if candidate.exists():
                os.add_dll_directory(str(candidate))
                path_val = os.environ.get("PATH", "")
                if str(candidate) not in path_val:
                    os.environ["PATH"] = f"{str(candidate)};{path_val}"
                print(f"  ✓ Added: {rel}")

# ===== 第 1 步：导入应用 =====
print("\n[1] 导入 app 模块...")
os.environ["MODEL_PROVIDER"] = "ollama"
os.environ["LOG_LEVEL"] = "INFO"
import app

# ===== 第 2 步：检查 ONNX Runtime CUDA 支持 =====
print("\n[2] 检查 ORT CUDA provider...")
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"   Available: {providers}")
has_cuda = 'CUDAExecutionProvider' in providers
print(f"   CUDA available: {'✓' if has_cuda else '✗'}")

# ===== 第 3 步：初始化人脸识别（最重要的部分）=====
print("\n[3] 初始化人脸识别...")
from app.services.face_recognition import FaceRecognitionService
face_service = FaceRecognitionService()
if face_service._app:
    print("   ✓ 人脸识别模型已加载")
    # 显示实际使用的 provider（会在日志中看到）
else:
    print("   ✗ 人脸识别模型加载失败")

# ===== 第 4 步：简单推理测试 =====
print("\n[4] 推理测试...")
try:
    import numpy as np
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    if face_service._app:
        faces = face_service._app.get(test_image)
        print(f"   ✓ 推理完成（人脸数: {len(faces)}）")
    else:
        print("   ✗ 人脸识别未初始化")
except Exception as e:
    print(f"   ✗ 推理失败: {e}")

print("\n完成！检查上方日志中的 'Applied providers:' 来判断是否使用了 GPU")
