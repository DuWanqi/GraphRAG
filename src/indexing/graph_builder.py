"""
GraphRAG知识图谱构建模块
负责将历史事件数据构建为知识图谱索引

使用 Microsoft GraphRAG: https://github.com/microsoft/graphrag
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import json

from ..config import get_settings


@dataclass
class IndexingResult:
    """索引构建结果"""
    success: bool
    message: str
    stats: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None


class GraphBuilder:
    """
    GraphRAG知识图谱构建器
    
    负责：
    1. 配置GraphRAG参数
    2. 处理历史事件数据
    3. 构建知识图谱索引
    4. 生成社区报告和摘要
    """
    
    # 历史事件专用实体类型
    ENTITY_TYPES = [
        "历史事件",
        "人物",
        "地点",
        "时间",
        "组织",
        "政策",
        "社会现象",
        "经济指标",
        "文化运动",
    ]
    
    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        llm_provider: str = "deepseek",
        llm_model: Optional[str] = None,
    ):
        """
        初始化图谱构建器
        
        Args:
            input_dir: 输入数据目录
            output_dir: 输出索引目录
            llm_provider: LLM提供商
            llm_model: LLM模型名称
        """
        settings = get_settings()
        
        self.input_dir = Path(input_dir or settings.graphrag_input_dir)
        self.output_dir = Path(output_dir or settings.graphrag_output_dir)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
        # 确保目录存在
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_llm_config(self) -> Dict[str, Any]:
        """
        获取LLM配置
        根据选择的提供商返回对应的配置
        
        Returns:
            包含以下字段的字典:
            - api_key: API密钥
            - env_var_name: 环境变量名称（GraphRAG用于读取密钥）
            - model_provider: GraphRAG model_provider 值
            - chat_model: 聊天模型名称
            - embedding_model: 嵌入模型名称
        """
        settings = get_settings()
        
        provider_configs = {
            "deepseek": {
                "api_key": settings.deepseek_api_key,
                "env_var_name": "DEEPSEEK_API_KEY",
                "model_provider": "deepseek",
                "chat_model": self.llm_model or "deepseek-chat",
                "embedding_model": "text-embedding-3-small",  # DeepSeek 用 OpenAI embedding
                "embedding_provider": "openai",
            },
            "qwen": {
                "api_key": settings.qwen_api_key,
                "env_var_name": "DASHSCOPE_API_KEY",
                "model_provider": "openai",  # 通过 OpenAI 兼容接口
                "api_base": settings.qwen_api_base,
                "chat_model": self.llm_model or "qwen-plus",
                "embedding_model": "text-embedding-v3",
                "embedding_provider": "openai",
            },
            "hunyuan": {
                "api_key": settings.hunyuan_api_key,
                "env_var_name": "HUNYUAN_API_KEY",
                "model_provider": "openai",  # 混元通过 OpenAI 兼容接口
                "chat_model": self.llm_model or "hunyuan-lite",
                "embedding_model": "text-embedding-3-small",
                "embedding_provider": "openai",
            },
            "gemini": {
                "api_key": settings.google_api_key,
                "env_var_name": "GEMINI_API_KEY",
                "model_provider": "gemini",
                "chat_model": self.llm_model or "gemini-2.0-flash",
                "embedding_model": "text-embedding-004",
                "embedding_provider": "gemini",
            },
            "openai": {
                "api_key": settings.openai_api_key,
                "env_var_name": "OPENAI_API_KEY",
                "model_provider": "openai",
                "chat_model": self.llm_model or "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "embedding_provider": "openai",
            },
        }
        
        return provider_configs.get(self.llm_provider, provider_configs["gemini"])
    
    def create_settings_yaml(self) -> str:
        """
        创建GraphRAG 2.x 配置文件
        根据选择的 LLM 提供商动态生成配置
        
        Returns:
            配置文件路径
        """
        llm_config = self._get_llm_config()
        env_var_name = llm_config.get("env_var_name", "GRAPHRAG_API_KEY")
        
        # 输入目录（相对于 output_dir）
        input_dir_rel = "input"
        input_dir_abs = self.output_dir / input_dir_rel
        input_dir_abs.mkdir(parents=True, exist_ok=True)
        
        # 复制输入文件到 output_dir/input
        for txt_file in self.input_dir.glob("*.txt"):
            dest = input_dir_abs / txt_file.name
            if not dest.exists():
                dest.write_text(txt_file.read_text(encoding="utf-8"), encoding="utf-8")
        
        # GraphRAG 2.x YAML 配置
        yaml_content = f"""### GraphRAG 2.x 配置文件
### 由记忆图谱系统自动生成
### LLM 提供商: {self.llm_provider}

models:
  default_chat_model:
    type: chat
    model_provider: {llm_config['model_provider']}
    auth_type: api_key
    api_key: ${{{env_var_name}}}
    model: {llm_config['chat_model']}
    model_supports_json: true
    concurrent_requests: 5
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 3
    tokens_per_minute: null
    requests_per_minute: null
  default_embedding_model:
    type: embedding
    model_provider: {llm_config.get('embedding_provider', llm_config['model_provider'])}
    auth_type: api_key
    api_key: ${{{env_var_name}}}
    model: {llm_config['embedding_model']}
    concurrent_requests: 5
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 3
    tokens_per_minute: null
    requests_per_minute: null

### Input settings ###

input:
  storage:
    type: file
    base_dir: "{input_dir_rel}"
  file_type: text

chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]

### Output/storage settings ###

output:
  type: file
  base_dir: "output"
    
cache:
  type: file
  base_dir: "cache"

reporting:
  type: file
  base_dir: "logs"

vector_store:
  default_vector_store:
    type: lancedb
    db_uri: output/lancedb
    container_name: default

### Workflow settings ###

embed_text:
  model_id: default_embedding_model
  vector_store_id: default_vector_store

extract_graph:
  model_id: default_chat_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization, person, geo, event, 历史事件, 人物, 地点, 时间, 组织]
  max_gleanings: 1

summarize_descriptions:
  model_id: default_chat_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

extract_graph_nlp:
  text_analyzer:
    extractor_type: regex_english
  async_mode: threaded

cluster_graph:
  max_cluster_size: 10

extract_claims:
  enabled: false
  model_id: default_chat_model
  prompt: "prompts/extract_claims.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 1

community_reports:
  model_id: default_chat_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 2000
  max_input_length: 8000

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: false
  embeddings: false

### Query settings ###

local_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  chat_model_id: default_chat_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

drift_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/drift_search_system_prompt.txt"
  reduce_prompt: "prompts/drift_search_reduce_prompt.txt"

basic_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/basic_search_system_prompt.txt"
"""
        
        # 写入配置文件
        settings_path = self.output_dir / "settings.yaml"
        settings_path.write_text(yaml_content.strip(), encoding="utf-8")
        
        return str(settings_path)
    
    def create_prompts(self):
        """
        创建自定义提示词文件
        针对历史事件场景优化
        """
        prompts_dir = self.output_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # 实体抽取提示词
        entity_extraction_prompt = """
-目标-
给定一段可能与历史事件相关的文本，识别出文本中所有的实体及其关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称
- entity_type: 实体类型，必须是以下之一：[历史事件, 人物, 地点, 时间, 组织, 政策, 社会现象, 经济指标, 文化运动]
- entity_description: 实体的详细描述，包括其历史背景和重要性

2. 从步骤1中识别的实体中，识别所有明显相关的(source_entity, target_entity)对。
对于每对相关实体，提取以下信息：
- source_entity: 源实体名称
- target_entity: 目标实体名称  
- relationship_description: 解释为什么源实体和目标实体相互关联
- relationship_strength: 表示源实体和目标实体之间关系强度的数值（1-10）
- relationship_keywords: 一个或多个高级关键词，用于总结关系的性质，例如：发生于、影响、引发、导致、促进、阻碍等

3. 返回输出为JSON格式，包含两个列表：
- "entities": 实体列表
- "relationships": 关系列表

-实际数据-
######################
文本: {input_text}
######################
输出:
"""
        
        # 摘要生成提示词
        summarize_prompt = """
你是一位历史学家，负责总结给定实体的历史背景和重要性。

给定一个或两个实体的描述列表，请将所有描述合并为一个综合性的历史描述。
确保包含所有描述中收集的信息，特别是时间、地点、人物和事件之间的关联。

描述应该：
1. 准确反映历史事实
2. 突出事件的时代背景
3. 说明与其他历史事件的关联
4. 使用客观、学术的语言风格

如果提供的描述存在矛盾，请解决矛盾并提供一个单一、连贯的摘要。

#############################
实体描述列表：
{description_list}
#############################
输出:
"""
        
        # 社区报告提示词
        community_report_prompt = """
你是一位历史分析专家。你将获得一组历史实体及其关系数据。
请撰写一份关于这组数据所代表的历史社区的综合报告。

报告应包含以下部分：

## 标题
为这个历史事件群起一个描述性的标题

## 摘要
总结这个历史社区的主要特征和重要性（2-3句话）

## 关键历史事件
列出并简要描述社区中的主要历史事件

## 重要人物
列出社区中的关键历史人物及其角色

## 时代背景
描述这些事件发生的时代背景和社会环境

## 历史影响
分析这些事件的历史影响和长远意义

## 关联分析
说明社区内各实体之间的关联和相互影响

请确保报告内容准确、客观，适合用于个人回忆录的历史背景补充。

#############################
实体数据：
{entities}

关系数据：
{relationships}
#############################
输出:
"""
        
        # 写入提示词文件
        (prompts_dir / "entity_extraction.txt").write_text(
            entity_extraction_prompt.strip(), encoding="utf-8"
        )
        (prompts_dir / "summarize_descriptions.txt").write_text(
            summarize_prompt.strip(), encoding="utf-8"
        )
        (prompts_dir / "community_report.txt").write_text(
            community_report_prompt.strip(), encoding="utf-8"
        )
    
    def prepare_input_data(self, data: List[Dict[str, Any]]) -> int:
        """
        准备输入数据
        将历史事件数据转换为GraphRAG可处理的文本格式
        
        Args:
            data: 历史事件数据列表，每项应包含：
                  - title: 事件标题
                  - date: 日期/时间
                  - location: 地点
                  - content: 事件内容描述
                  
        Returns:
            处理的文件数量
        """
        file_count = 0
        
        for i, item in enumerate(data):
            # 构建文本内容
            text_parts = []
            
            if "title" in item:
                text_parts.append(f"事件标题：{item['title']}")
            if "date" in item:
                text_parts.append(f"时间：{item['date']}")
            if "location" in item:
                text_parts.append(f"地点：{item['location']}")
            if "content" in item:
                text_parts.append(f"\n{item['content']}")
            
            text = "\n".join(text_parts)
            
            # 写入文件
            filename = f"event_{i:04d}.txt"
            filepath = self.input_dir / filename
            filepath.write_text(text, encoding="utf-8")
            file_count += 1
        
        return file_count
    
    def _create_env_file(self, llm_config: Dict[str, Any]) -> None:
        """
        创建 GraphRAG 所需的 .env 文件
        """
        env_var_name = llm_config.get("env_var_name", "GRAPHRAG_API_KEY")
        api_key = llm_config.get("api_key", "")
        
        env_content = f"{env_var_name}={api_key}"
        env_path = self.output_dir / ".env"
        
        # 使用 UTF-8 编码（无 BOM）
        env_path.write_text(env_content, encoding="utf-8")
    
    async def build_index(self) -> IndexingResult:
        """
        构建知识图谱索引
        
        Returns:
            IndexingResult: 构建结果
        """
        try:
            # 检查输入数据
            input_files = list(self.input_dir.glob("*.txt"))
            if not input_files:
                return IndexingResult(
                    success=False,
                    message="输入目录中没有找到.txt文件",
                    output_dir=str(self.output_dir)
                )
            
            # 获取 LLM 配置
            llm_config = self._get_llm_config()
            
            # 检查 API 密钥
            if not llm_config.get("api_key"):
                return IndexingResult(
                    success=False,
                    message=f"未配置 {self.llm_provider} 的 API 密钥，请在 .env 文件中设置",
                    output_dir=str(self.output_dir)
                )
            
            # 初始化 GraphRAG（如果 prompts 目录不存在）
            prompts_dir = self.output_dir / "prompts"
            if not prompts_dir.exists():
                init_result = subprocess.run(
                    [sys.executable, "-m", "graphrag", "init", "--root", str(self.output_dir.resolve())],
                    capture_output=True,
                    text=True,
                    cwd=str(self.output_dir.resolve()),
                )
                if init_result.returncode != 0:
                    return IndexingResult(
                        success=False,
                        message=f"GraphRAG 初始化失败: {init_result.stderr}",
                        output_dir=str(self.output_dir)
                    )
            
            # 创建配置文件（会覆盖默认配置）
            settings_path = self.create_settings_yaml()
            
            # 创建 .env 文件
            self._create_env_file(llm_config)
            
            # 设置环境变量
            env_var_name = llm_config.get("env_var_name", "GRAPHRAG_API_KEY")
            env = {**os.environ, env_var_name: llm_config["api_key"]}
            
            # 运行 GraphRAG 索引管道
            result = subprocess.run(
                [
                    sys.executable, "-m", "graphrag", "index",
                    "--root", str(self.output_dir.resolve()),
                ],
                capture_output=True,
                text=True,
                cwd=str(self.output_dir.resolve()),
                env=env
            )
            
            if result.returncode == 0:
                return IndexingResult(
                    success=True,
                    message="知识图谱索引构建成功",
                    stats={
                        "input_files": len(input_files),
                        "output_dir": str(self.output_dir),
                        "llm_provider": self.llm_provider,
                        "model": llm_config.get("chat_model", "unknown"),
                    },
                    output_dir=str(self.output_dir)
                )
            else:
                error_msg = result.stderr[:500] if result.stderr else "未知错误"
                return IndexingResult(
                    success=False,
                    message=f"索引构建失败: {error_msg}",
                    output_dir=str(self.output_dir)
                )
                
        except Exception as e:
            return IndexingResult(
                success=False,
                message=f"索引构建异常: {str(e)}",
                output_dir=str(self.output_dir)
            )
    
    def build_index_sync(self) -> IndexingResult:
        """同步版本的索引构建"""
        return asyncio.run(self.build_index())
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            包含节点数、边数等统计信息的字典
        """
        stats = {
            "indexed": False,
            "input_files": 0,
            "entities": 0,
            "relationships": 0,
            "communities": 0,
        }
        
        # 检查输入文件（两个可能的位置）
        input_files = list(self.input_dir.glob("*.txt"))
        input_files_alt = list((self.output_dir / "input").glob("*.txt"))
        stats["input_files"] = max(len(input_files), len(input_files_alt))
        
        # GraphRAG 2.x 输出文件位置
        output_dir = self.output_dir / "output"
        
        # 检查实体文件（GraphRAG 2.x 格式）
        entities_file = output_dir / "entities.parquet"
        if not entities_file.exists():
            # 兼容旧版格式
            entities_file = output_dir / "create_final_entities.parquet"
        
        if entities_file.exists():
            stats["indexed"] = True
            try:
                import pandas as pd
                entities_df = pd.read_parquet(entities_file)
                stats["entities"] = len(entities_df)
            except Exception:
                pass
        
        # 检查关系文件
        relationships_file = output_dir / "relationships.parquet"
        if not relationships_file.exists():
            relationships_file = output_dir / "create_final_relationships.parquet"
        
        if relationships_file.exists():
            try:
                import pandas as pd
                rel_df = pd.read_parquet(relationships_file)
                stats["relationships"] = len(rel_df)
            except Exception:
                pass
        
        # 检查社区文件
        communities_file = output_dir / "communities.parquet"
        if not communities_file.exists():
            communities_file = output_dir / "create_final_communities.parquet"
        
        if communities_file.exists():
            try:
                import pandas as pd
                comm_df = pd.read_parquet(communities_file)
                stats["communities"] = len(comm_df)
            except Exception:
                pass
        
        return stats
