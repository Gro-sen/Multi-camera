#!/usr/bin/env python
"""测试 Ollama 模型 GPU 使用情况"""
import time
import os
os.environ["MODEL_PROVIDER"] = "ollama"

print("=" * 60)
print("Ollama 模型 GPU 加速测试")
print("=" * 60)

from app.models.ollama_vision import OllamaVisionModel
from app.models.ollama_reasoning import OllamaReasoningModel

# 测试视觉模型
print("\n[1] 初始化 qwen3-vl:4b 视觉模型...")
try:
    vision_model = OllamaVisionModel()
    print("    ✓ 模型初始化完成")
except Exception as e:
    print(f"    ✗ 失败: {e}")
    exit(1)

# 测试推理模型
print("\n[2] 初始化 qwen2.5:7b 推理模型...")
try:
    reasoning_model = OllamaReasoningModel()
    print("    ✓ 模型初始化完成")
except Exception as e:
    print(f"    ✗ 失败: {e}")
    exit(1)

# 测试视觉推理耗时
print("\n[3] 测试视觉推理 (10字符提示)...")
import numpy as np
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
test_prompt = "人员检测"

start = time.time()
try:
    result = vision_model.infer(test_image, {"question": test_prompt})
    elapsed = time.time() - start
    print(f"    ✓ 完成，耗时: {elapsed:.2f}s")
    print(f"    结果长度: {len(result) if result else 0} 字符")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试推理模型耗时
print("\n[4] 测试推理模型 (简短询问)...")
test_query = "这个人是谁？"

start = time.time()
try:
    result = reasoning_model.infer(test_query, [])
    elapsed = time.time() - start
    print(f"    ✓ 完成，耗时: {elapsed:.2f}s")
    print(f"    结果长度: {len(str(result))} 字符")
except Exception as e:
    print(f"    ✗ 失败: {e}")

print("\n" + "=" * 60)
print("测试完成。")
print("注意：查看 nvidia-smi 或 Ollama 日志确认是否使用了 GPU。")
print("如果耗时 >30s 且 nvidia-smi 无 GPU 占用，则 Ollama 未启用 GPU。")
print("=" * 60)
