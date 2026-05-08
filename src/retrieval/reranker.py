"""
重排序模块
使用 BCE-Reranker 对检索结果进行重排序
支持中英混合场景
"""

from typing import List, Dict, Any, Optional
import os


class Reranker:
    """
    BCE-Reranker 重排序器
    
    使用网易有道的 BCE-Reranker 模型对检索结果进行重排序
    特别优化中英混合场景
    """
    
    def __init__(self, model_name: str = "maidalun1020/bce-reranker-base_v1"):
        """
        初始化重排序器
        
        Args:
            model_name: 模型名称，默认使用 BCE-Reranker
        """
        self.model_name = model_name
        self._model = None
        self._error_message = None
        self._load_model()
    
    def _load_model(self):
        """加载重排序模型"""
        try:
            from sentence_transformers import CrossEncoder
            import os
            
            # 设置 HuggingFace 镜像端点（国内加速）
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            print(f"[Reranker] 正在加载模型: {self.model_name}")
            print(f"[Reranker] 使用镜像: https://hf-mirror.com")
            self._model = CrossEncoder(self.model_name)
            print(f"[Reranker] 模型加载成功")
            self._error_message = None
        except Exception as e:
            error_str = str(e)
            print(f"[Reranker] 模型加载失败: {e}")
            print(f"[Reranker] 请安装依赖: pip install sentence-transformers")
            self._model = None
            if "Can't load the model" in error_str:
                self._error_message = (
                    f"重排序模型下载失败。请检查网络连接或手动下载模型:\n"
                    f"模型: {self.model_name}\n"
                    f"手动下载命令: `huggingface-cli download {self.model_name}`"
                )
            else:
                self._error_message = f"重排序模型加载失败: {error_str[:100]}"
    
    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        return self._model is not None
    
    def get_error_message(self) -> Optional[str]:
        """获取错误信息"""
        return self._error_message
    
    def rerank_entities(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        对实体进行重排序
        
        使用混合策略：结合原始检索分数和重排序分数
        对于短查询，更依赖原始检索分数
        
        Args:
            query: 查询文本
            entities: 实体列表
            top_k: 返回前k个结果
            
        Returns:
            重排序后的实体列表
        """
        if not self.is_ready() or not entities:
            return entities[:top_k]
        
        # 构建查询-文档对（只使用实体名称，不使用长描述）
        pairs = []
        for entity in entities:
            name = entity.get("name", entity.get("title", ""))
            # 只使用名称进行重排序，避免长描述干扰
            pairs.append([query, name])
        
        # 获取重排序分数
        rerank_scores = self._model.predict(pairs)
        
        # 计算混合分数
        for i, entity in enumerate(entities):
            # 原始分数（归一化到 0-1）
            original_score = entity.get("score", 0.5)
            if original_score > 1:
                original_score = 1.0
            
            # 重排序分数（sigmoid 归一化）
            rerank_score = float(rerank_scores[i])
            # BCE-Reranker 输出的是 logits，需要 sigmoid 转换
            import math
            rerank_score_norm = 1 / (1 + math.exp(-rerank_score))
            
            # 混合分数：原始分数权重 0.6，重排序分数权重 0.4
            # 对于短查询，更信任原始检索
            hybrid_score = 0.6 * original_score + 0.4 * rerank_score_norm
            
            entity["rerank_score"] = rerank_score_norm
            entity["hybrid_score"] = hybrid_score
            entity["original_score"] = original_score
        
        # 按混合分数排序
        sorted_entities = sorted(
            entities,
            key=lambda x: x.get("hybrid_score", 0),
            reverse=True
        )
        
        return sorted_entities[:top_k]
    
    def rerank_relationships(
        self,
        query: str,
        relationships: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        对关系进行重排序
        
        使用混合策略：结合原始检索分数和重排序分数
        
        Args:
            query: 查询文本
            relationships: 关系列表
            top_k: 返回前k个结果
            
        Returns:
            重排序后的关系列表
        """
        if not self.is_ready() or not relationships:
            return relationships[:top_k]
        
        # 构建查询-文档对（简化关系表示）
        pairs = []
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "")
            # 只使用 source-target-type，不使用长描述
            doc_text = f"{source} {rel_type} {target}"
            pairs.append([query, doc_text])
        
        # 获取重排序分数
        rerank_scores = self._model.predict(pairs)
        
        # 计算混合分数
        for i, rel in enumerate(relationships):
            # 原始分数
            original_score = rel.get("score", 0.5)
            if original_score > 1:
                original_score = 1.0
            
            # 重排序分数（sigmoid 归一化）
            rerank_score = float(rerank_scores[i])
            import math
            rerank_score_norm = 1 / (1 + math.exp(-rerank_score))
            
            # 混合分数
            hybrid_score = 0.6 * original_score + 0.4 * rerank_score_norm
            
            rel["rerank_score"] = rerank_score_norm
            rel["hybrid_score"] = hybrid_score
            rel["original_score"] = original_score
        
        # 按混合分数排序
        sorted_relationships = sorted(
            relationships,
            key=lambda x: x.get("hybrid_score", 0),
            reverse=True
        )
        
        return sorted_relationships[:top_k]
    
    def rerank_text_units(
        self,
        query: str,
        text_units: List[str],
        top_k: int = 5,
    ) -> List[str]:
        """
        对文本单元进行重排序
        
        Args:
            query: 查询文本
            text_units: 文本单元列表
            top_k: 返回前k个结果
            
        Returns:
            重排序后的文本单元列表
        """
        if not self.is_ready() or not text_units:
            return text_units[:top_k]
        
        # 构建查询-文档对
        pairs = [[query, text] for text in text_units]
        
        # 获取重排序分数
        scores = self._model.predict(pairs)
        
        # 按分数排序
        scored_texts = list(zip(text_units, scores))
        scored_texts.sort(key=lambda x: x[1], reverse=True)
        
        return [text for text, _ in scored_texts[:top_k]]


# 全局重排序器实例
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """获取全局重排序器实例"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
