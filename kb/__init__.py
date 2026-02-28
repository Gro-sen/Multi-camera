"""
知识库模块 - 集成版本
提供统一的知识库接口
"""
import os
import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from app.core import get_logger

logger = get_logger(__name__)


class KnowledgeBase:
    """知识库管理类"""
    
    def __init__(self, base_dir: str = "kb"):
        """初始化知识库
        
        Args:
            base_dir: 知识库基础目录
        """
        # 使用 Path 获取绝对路径
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
        
        self.base_dir = str(base_path)
        self.source_dir = os.path.join(self.base_dir, "source")
        self.index_dir = os.path.join(self.base_dir, "index")
        self.cases_dir = os.path.join(self.base_dir, "cases")
        
        # 创建必要的目录
        for dir_path in [self.source_dir, self.index_dir, self.cases_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.debug(f"知识库已初始化: {self.base_dir}")
    
    def add_case(self, case_data: Dict[str, Any]) -> str:
        """添加报警案例到知识库
        
        Args:
            case_data: 案例数据
            
        Returns:
            案例ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_id = f"case_{timestamp}"
        
        case_file = os.path.join(self.cases_dir, f"{case_id}.json")
        
        # 添加元数据
        case_data.update({
            "case_id": case_id,
            "created_at": datetime.now().isoformat(),
            "reviewed": False,
            "review_result": None
        })
        
        with open(case_file, 'w', encoding='utf-8') as f:
            json.dump(case_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"案例已保存: {case_id}")
        
        # 同时写入Markdown格式（供索引）
        try:
            from .auto_writer import write_alarm_case_to_kb
            write_alarm_case_to_kb(case_data)
        except Exception as e:
            logger.warning(f"写入自动文件失败: {e}")
        
        return case_id
    
    def get_similar_cases(self, query_text: str, top_k: int = 3, similarity_threshold: float = 0.3) -> List[Dict]:
        """获取相似案例
        
        Args:
            query_text: 查询文本
            top_k: 返回数量
            similarity_threshold: 相似度阈值
            
        Returns:
            相似案例列表
        """
        try:
            from .retriever import query
            
            results = query(query_text, top_k=top_k, similarity_threshold=similarity_threshold)
            
            # 转换为统一格式
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.get("text", ""),
                    "source": result.get("source", ""),
                    "score": result.get("score", 0.0),
                    "metadata": {
                        "chunk_type": "rule_chunk",
                        "retrieved_at": datetime.now().isoformat()
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"知识库查询失败: {e}")
            return []
    
    def update_index(self) -> Dict[str, Any]:
        """更新知识库索引
        
        Returns:
            更新结果
        """
        try:
            from .indexing import build_index
            result = build_index(
                data_dir=self.source_dir,
                index_path=os.path.join(self.index_dir, "faiss_bge.index"),
                meta_path=os.path.join(self.index_dir, "docs_bge.pkl")
            )
            return result
        except Exception as e:
            logger.error(f"索引更新失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_cases": 0,
            "total_documents": 0,
            "index_exists": False,
            "last_update": None,
            "status": "ready"
        }
        
        try:
            # 计算案例数
            if os.path.exists(self.cases_dir):
                stats["total_cases"] = len([
                    f for f in os.listdir(self.cases_dir) 
                    if f.endswith('.json')
                ])
            
            # 计算文档数
            if os.path.exists(self.source_dir):
                stats["total_documents"] = len([
                    f for f in os.listdir(self.source_dir) 
                    if f.endswith('.md')
                ])
            
            # 检查索引
            index_file = os.path.join(self.index_dir, "faiss_bge.index")
            stats["index_exists"] = os.path.exists(index_file)
            
            if stats["index_exists"]:
                stats["last_update"] = datetime.fromtimestamp(
                    os.path.getmtime(index_file)
                ).strftime("%Y-%m-%d %H:%M:%S")
            
        except Exception as e:
            logger.warning(f"获取统计信息失败: {e}")
            stats["status"] = "error"
        
        return stats
    
    def check_index_health(self) -> Dict[str, Any]:
        """检查索引健康状况
        
        Returns:
            健康状态信息
        """
        try:
            index_file = os.path.join(self.index_dir, "faiss_bge.index")
            meta_file = os.path.join(self.index_dir, "docs_bge.pkl")
            
            if not os.path.exists(index_file) or not os.path.exists(meta_file):
                return {
                    "status": "missing",
                    "message": "索引文件不存在"
                }
            
            from .retriever import load_index
            index, meta, model = load_index(
                index_path=index_file,
                meta_path=meta_file
            )
            
            return {
                "status": "healthy",
                "index_size": index.ntotal,
                "meta_count": len(meta) if meta else 0
            }
            
        except Exception as e:
            return {
                "status": "corrupted",
                "message": str(e)
            }


# 创建全局实例
kb = KnowledgeBase(base_dir="kb")