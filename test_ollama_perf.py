#!/usr/bin/env python
"""Ollama GPU 加速测试 - 正确方法签名"""
import time
import base64
import os
os.environ["MODEL_PROVIDER"] = "ollama"

print("=" * 60)
print("Ollama 模型推理速度测试")
print("=" * 60)

from app.models.ollama_vision import OllamaVisionModel
from app.models.ollama_reasoning import OllamaReasoningModel
import numpy as np

# 创建测试图像（黑色640x480）
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
ret, buf = __import__('cv2').imencode('.jpg', test_image)
image_b64 = base64.b64encode(buf).decode('utf-8')

# 初始化模型
print("\n[1] 初始化 Ollama 模型...")
try:
    vision_model = OllamaVisionModel()
    reasoning_model = OllamaReasoningModel()
    print("    ✓ 模型已初始化")
except Exception as e:
    print(f"    ✗ 初始化失败: {e}")
    exit(1)

# 测试视觉推理
print("\n[2] 测试视觉推理 (qwen3-vl:4b)...")
test_prompt = "画中有人吗？"
start = time.time()
try:
    result = vision_model.analyze(image_b64, test_prompt)
    elapsed = time.time() - start
    print(f"    ✓ 完成，耗时: {elapsed:.2f}s")
    print(f"    结果: {result[:50]}..." if result else "    结果: (空)")
except Exception as e:
    print(f"    ✗ 失败: {e}")
    elapsed = None

# 测试推理模型
print("\n[3] 测试推理模型 (qwen2.5:7b)...")
facts = {"person_count": 1, "has_badge": False}
cases = []
prompt = "需要报警吗？"
start = time.time()
try:
    result = reasoning_model.infer(facts, cases, prompt)
    elapsed = time.time() - start
    print(f"    ✓ 完成，耗时: {elapsed:.2f}s")
    print(f"    结果: {str(result)[:50]}..." if result else "    结果: (空)")
except Exception as e:
    print(f"    ✗ 失败: {e}")
    elapsed = None

print("\n" + "=" * 60)
print("✓ 测试完成")
print("\nGPU 加速状态判断：")
print("  - 快速 (<10s): ✓ GPU 已启用")
print("  - 中等 (10-30s): △ GPU 可能部分启用")
print("  - 缓慢 (>30s): ✗ 未使用 GPU")
print("\n运行此脚本的同时查看 Task Manager 中 GPU 的使用情况。")
print("=" * 60)
