"""
重建GraphRAG索引（使用Ollama embedding）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import GraphBuilder

print("=" * 60)
print("重建GraphRAG索引（使用Ollama embedding）")
print("=" * 60)

# 创建GraphBuilder，使用Ollama embedding
builder = GraphBuilder(
    llm_provider="gemini",  # LLM仍用Gemini
    embedding_provider="ollama",  # Embedding用Ollama
    embedding_model="nomic-embed-text",
)

print(f"\n输入目录: {builder.input_dir}")
print(f"输出目录: {builder.output_dir}")
print(f"LLM提供商: {builder.llm_provider}")
print(f"Embedding提供商: {builder.embedding_provider}")
print(f"Embedding模型: {builder.embedding_model}")

# 创建配置文件
print("\n创建配置文件...")
settings_path = builder.create_settings_yaml()
print(f"配置文件: {settings_path}")

# 构建索引
print("\n开始构建索引...")
result = builder.build_index_sync()

if result.success:
    print("\n" + "=" * 60)
    print("索引构建成功！")
    print(f"输出目录: {result.output_dir}")
    print(f"统计信息: {result.stats}")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print(f"索引构建失败: {result.message}")
    print("=" * 60)
