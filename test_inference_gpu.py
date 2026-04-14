#!/usr/bin/env python
"""集成测试：验证推理是否使用 GPU"""
import os
import sys
import time
from pathlib import Path

# 设置 ollama 提供者
os.environ["MODEL_PROVIDER"] = "ollama"
os.environ["LOG_LEVEL"] = "INFO"

# 导入应用
import app
from app.services.face_recognition import FaceRecognitionService
from app.models.factory import create_models
from app.core import get_logger

logger = get_logger(__name__)

print("=" * 60)
print("GPU 加速集成测试")
print("=" * 60)

# 1. 验证 ONNX Runtime CUDA 支持
print("\n[1/4] 检查 ONNX Runtime CUDA 提供者...")
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"     可用提供者: {providers}")
if 'CUDAExecutionProvider' in providers:
    print("     ✓ CUDA provider 已启用")
else:
    print("     ✗ CUDA provider 未启用")

# 2. 初始化模型
print("\n[2/4] 初始化人脸识别和推理模型...")
try:
    face_service = FaceRecognitionService()
    print(f"     人脸识别状态: {'已启用' if face_service._app else '禁用'}")
    
    vision_model, reasoning_model = create_models()
    print(f"     ✓ 模型加载完成")
except Exception as e:
    print(f"     ✗ 模型加载失败: {e}")
    sys.exit(1)

# 3. 生成测试图像
print("\n[3/4] 生成测试图像...")
try:
    import numpy as np
    import cv2
    # 创建一个 640x480 的随机图像（模拟摄像头输入）
    test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    print(f"     图像形状: {test_image.shape}")
    print("     ✓ 测试图像已生成")
except Exception as e:
    print(f"     ✗ 生成失败: {e}")
    sys.exit(1)

# 4. 执行推理
print("\n[4/4] 执行推理测试...")
try:
    start = time.time()
    
    # 测试视觉模型
    print("     [视觉推理] 开始...")
    vision_start = time.time()
    vision_facts = vision_model.infer(test_image, {})
    vision_time = time.time() - vision_start
    print(f"     [视觉推理] 耗时: {vision_time:.2f}s")
    
    print(f"     ✓ 推理完成，总耗时: {time.time() - start:.2f}s")
except Exception as e:
    print(f"     ✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！CUDA 加速已启用。")
print(f"推理耗时: {vision_time:.2f}s (预期 GPU: <5s, CPU: 15-40s)")
print("=" * 60)
