"""
Vision Describe - 智能安防视频分析系统
"""
__version__ = "1.0.0"

# 在任何其他导入之前，确保 CUDA DLL 搜索路径已设置（Windows）
def _bootstrap_cuda_on_windows():
    """在 Windows 上尽早将 NVIDIA CUDA DLL 目录加入搜索路径（必须在导入 ONNX Runtime 之前）"""
    import os
    import site
    from pathlib import Path
    import sys

    if os.name != "nt":
        return

    try:
        site_paths = site.getsitepackages()
    except Exception:
        site_paths = []

    dll_dirs = []
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
                dll_dirs.append(str(candidate))
                try:
                    os.add_dll_directory(str(candidate))
                except Exception:
                    pass

    # 关键：也要更新 PATH 环境变量，确保 DLL 在所有环境中都能被找到
    if dll_dirs:
        existing_path = os.environ.get("PATH", "")
        for dll_dir in reversed(dll_dirs):
            if dll_dir not in existing_path:
                existing_path = f"{dll_dir};{existing_path}" if existing_path else dll_dir
        os.environ["PATH"] = existing_path
        
        # 仅在启动时记录一次（避免重复日志）
        if not hasattr(sys, "_cuda_bootstrap_logged"):
            print(f"[CUDA BOOTSTRAP] Injected {len(dll_dirs)} NVIDIA DLL paths", file=sys.stderr)
            sys._cuda_bootstrap_logged = True

_bootstrap_cuda_on_windows()