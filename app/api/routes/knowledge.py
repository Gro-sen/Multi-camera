"""
知识库相关路由
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Tuple
from pathlib import Path
import shutil

from app.core import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


def _get_kb_dirs() -> Tuple[Path, Path]:
    from kb import kb
    base_dir = Path(kb.base_dir)
    pend_dir = base_dir / "pend"
    source_dir = base_dir / "source"
    pend_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    return pend_dir, source_dir


@router.get("/stats")
async def get_kb_stats():
    """获取知识库统计信息"""
    try:
        from kb import kb
        stats = kb.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"获取知识库统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取知识库统计失败")


@router.get("/search")
async def search_cases(
    query: str,
    top_k: int = Query(5, ge=1, le=20),
    threshold: float = Query(0.3, ge=0.0, le=1.0)
):
    """搜索相似案例"""
    try:
        from kb import kb
        results = kb.get_similar_cases(query, top_k, threshold)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"知识库搜索失败: {e}")
        raise HTTPException(status_code=500, detail="知识库搜索失败")


@router.post("/add-case")
async def add_case(case_data: dict):
    """添加新的报警案例"""
    try:
        required_fields = ["scene_summary", "alarm_level", "alarm_reason"]
        for field in required_fields:
            if field not in case_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"缺少必要字段: {field}"
                )
        
        from kb import kb
        case_id = kb.add_case(case_data)
        
        return {
            "status": "success",
            "case_id": case_id,
            "message": "案例已添加"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加案例失败: {e}")
        raise HTTPException(status_code=500, detail="添加案例失败")


@router.post("/update-index")
async def update_kb_index():
    """更新知识库索引"""
    try:
        from kb import kb
        result = kb.update_index()
        return {
            "status": "success",
            "message": "知识库索引已更新",
            "result": result
        }
    except Exception as e:
        logger.error(f"更新知识库索引失败: {e}")
        raise HTTPException(status_code=500, detail="更新知识库索引失败")


@router.get("/pend-files")
async def list_pend_files():
    """列出待审核目录中的案例文件"""
    try:
        pend_dir, _ = _get_kb_dirs()
        files = []

        for p in sorted(pend_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = p.stat()
            case_type = "alarm" if p.name.startswith("alarm_") else "normal" if p.name.startswith("normal_") else "unknown"
            files.append({
                "name": p.name,
                "type": case_type,
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
            })

        return {
            "status": "success",
            "count": len(files),
            "files": files,
        }
    except Exception as e:
        logger.error(f"获取待审核文件失败: {e}")
        raise HTTPException(status_code=500, detail="获取待审核文件失败")


@router.get("/pend-preview")
async def get_pend_file_preview(name: str = Query(..., description="待审核文件名")):
    """预览 pend 目录中的 markdown 文件内容"""
    try:
        safe_name = Path(name).name
        if safe_name != name or not safe_name.lower().endswith(".md"):
            raise HTTPException(status_code=400, detail="非法文件名")

        pend_dir, _ = _get_kb_dirs()
        target = pend_dir / safe_name
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")

        content = target.read_text(encoding="utf-8")
        return {
            "status": "success",
            "name": safe_name,
            "content": content,
        }
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件编码不支持，需为 UTF-8")
    except Exception as e:
        logger.error(f"预览待审核文件失败: {e}")
        raise HTTPException(status_code=500, detail="预览待审核文件失败")


@router.post("/delete-pend")
async def delete_pend_files(payload: dict):
    """删除选中的 pend 文件"""
    try:
        files: List[str] = payload.get("files", [])
        if not files:
            raise HTTPException(status_code=400, detail="files 不能为空")

        pend_dir, _ = _get_kb_dirs()
        deleted = []
        skipped = []

        for file_name in files:
            safe_name = Path(file_name).name
            if safe_name != file_name or not safe_name.lower().endswith(".md"):
                skipped.append({"name": file_name, "reason": "非法文件名"})
                continue

            target = pend_dir / safe_name
            if not target.exists() or not target.is_file():
                skipped.append({"name": safe_name, "reason": "文件不存在"})
                continue

            target.unlink()
            deleted.append(safe_name)

        return {
            "status": "success",
            "deleted_count": len(deleted),
            "deleted": deleted,
            "skipped": skipped,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除待审核文件失败: {e}")
        raise HTTPException(status_code=500, detail="删除待审核文件失败")


@router.post("/import-pend")
async def import_pend_files(payload: dict):
    """将选中的 pend 文件导入 source（默认移动）"""
    try:
        files: List[str] = payload.get("files", [])
        move: bool = bool(payload.get("move", True))

        if not files:
            raise HTTPException(status_code=400, detail="files 不能为空")

        pend_dir, source_dir = _get_kb_dirs()
        imported = []
        skipped = []

        for file_name in files:
            safe_name = Path(file_name).name
            if safe_name != file_name or not safe_name.lower().endswith(".md"):
                skipped.append({"name": file_name, "reason": "非法文件名"})
                continue

            src = pend_dir / safe_name
            if not src.exists() or not src.is_file():
                skipped.append({"name": safe_name, "reason": "文件不存在"})
                continue

            dst = source_dir / safe_name
            if dst.exists():
                skipped.append({"name": safe_name, "reason": "source 中已存在同名文件"})
                continue

            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            imported.append(safe_name)

        return {
            "status": "success",
            "imported_count": len(imported),
            "imported": imported,
            "skipped": skipped,
            "mode": "move" if move else "copy",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导入待审核文件失败: {e}")
        raise HTTPException(status_code=500, detail="导入待审核文件失败")