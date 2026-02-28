"""
知识库相关路由
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.core import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


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