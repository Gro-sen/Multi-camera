"""
人脸白名单管理路由
"""
from pathlib import Path
from typing import List, Dict, Any
import shutil
import time

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.core import get_logger, config

logger = get_logger(__name__)

router = APIRouter(prefix="/faces", tags=["face-whitelist"])

_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _safe_name(name: str) -> str:
    return Path(name).name


def _identity_from_stem(stem: str) -> str:
    import re

    cleaned = stem.strip()
    if not cleaned:
        return stem
    merged = re.sub(r"(?:[_\-\s]+\d+)$", "", cleaned)
    return merged if merged else cleaned


def _scan_whitelist() -> Dict[str, Any]:
    whitelist_dir = Path(config.FACE_WHITELIST_DIR)
    whitelist_dir.mkdir(parents=True, exist_ok=True)

    files: List[Dict[str, Any]] = []
    identities: Dict[str, int] = {}

    for p in sorted(whitelist_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _ALLOWED_EXTS:
            continue

        identity = _identity_from_stem(p.stem)
        identities[identity] = identities.get(identity, 0) + 1
        stat = p.stat()
        files.append(
            {
                "name": p.name,
                "identity": identity,
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
            }
        )

    return {
        "directory": str(whitelist_dir),
        "files": files,
        "identities": [{"name": k, "templates": v} for k, v in sorted(identities.items())],
        "identity_count": len(identities),
        "file_count": len(files),
        "enabled": config.FACE_RECOGNITION_ENABLED,
        "threshold": config.FACE_MATCH_THRESHOLD,
    }


@router.get("/whitelist")
async def get_whitelist():
    """获取白名单文件与身份聚合信息"""
    try:
        data = _scan_whitelist()
        return {"status": "success", **data}
    except Exception as e:
        logger.error("获取白名单失败: %s", e)
        raise HTTPException(status_code=500, detail="获取白名单失败")


@router.post("/upload")
async def upload_whitelist_images(
    files: List[UploadFile] = File(...),
    identity: str = Form(""),
):
    """上传白名单图片"""
    try:
        whitelist_dir = Path(config.FACE_WHITELIST_DIR)
        whitelist_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        skipped = []
        prefix = identity.strip()
        ts = int(time.time())

        for idx, upload in enumerate(files, start=1):
            original_name = _safe_name(upload.filename or "")
            ext = Path(original_name).suffix.lower()
            if ext not in _ALLOWED_EXTS:
                skipped.append({"name": original_name or "(empty)", "reason": "不支持的图片格式"})
                continue

            if prefix:
                stem = f"{prefix}_{ts}_{idx}"
            else:
                stem = Path(original_name).stem.strip() or f"face_{ts}_{idx}"

            target = whitelist_dir / f"{stem}{ext}"
            with target.open("wb") as f:
                shutil.copyfileobj(upload.file, f)

            saved.append(target.name)

        return {
            "status": "success",
            "saved": saved,
            "saved_count": len(saved),
            "skipped": skipped,
        }
    except Exception as e:
        logger.error("上传白名单图片失败: %s", e)
        raise HTTPException(status_code=500, detail="上传白名单图片失败")


@router.post("/delete")
async def delete_whitelist_files(payload: dict):
    """删除白名单图片"""
    try:
        names = payload.get("files", [])
        if not names:
            raise HTTPException(status_code=400, detail="files 不能为空")

        whitelist_dir = Path(config.FACE_WHITELIST_DIR)
        deleted = []
        skipped = []

        for name in names:
            safe_name = _safe_name(str(name))
            if safe_name != name:
                skipped.append({"name": name, "reason": "非法文件名"})
                continue

            path = whitelist_dir / safe_name
            if not path.exists() or not path.is_file():
                skipped.append({"name": safe_name, "reason": "文件不存在"})
                continue

            if path.suffix.lower() not in _ALLOWED_EXTS:
                skipped.append({"name": safe_name, "reason": "不支持的文件类型"})
                continue

            path.unlink()
            deleted.append(safe_name)

        return {
            "status": "success",
            "deleted": deleted,
            "deleted_count": len(deleted),
            "skipped": skipped,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("删除白名单图片失败: %s", e)
        raise HTTPException(status_code=500, detail="删除白名单图片失败")


@router.post("/refresh")
async def refresh_whitelist_runtime():
    """刷新运行中人脸白名单缓存"""
    try:
        refreshed = 0

        # 推理工作线程中的各摄像头服务
        try:
            from app.main import lifecycle

            worker = getattr(lifecycle, "inference_worker", None)
            services = getattr(worker, "services", {}) if worker is not None else {}
            for _, service in services.items():
                face_service = getattr(service, "face_service", None)
                if face_service is not None:
                    face_service.refresh_whitelist()
                    refreshed += 1
        except Exception as e:
            logger.debug("刷新运行中服务失败: %s", e)

        data = _scan_whitelist()
        return {
            "status": "success",
            "message": "白名单已刷新",
            "refreshed_services": refreshed,
            "identity_count": data["identity_count"],
            "file_count": data["file_count"],
        }
    except Exception as e:
        logger.error("刷新白名单失败: %s", e)
        raise HTTPException(status_code=500, detail="刷新白名单失败")
