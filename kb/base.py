"""
知识库基类 - 可选的抽象层
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class KnowledgeBaseBase(ABC):
    """知识库抽象基类"""
    
    @abstractmethod
    def add_case(self, case_data: Dict[str, Any]) -> str:
        """添加案例"""
        pass
    
    @abstractmethod
    def get_similar_cases(self, query_text: str, top_k: int = 3, similarity_threshold: float = 0.3) -> List[Dict]:
        """获取相似案例"""
        pass
    
    @abstractmethod
    def update_index(self) -> Dict[str, Any]:
        """更新索引"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        pass