"""检查索引数据"""
import pandas as pd
from pathlib import Path

output_dir = Path("data/output/output")

entities_file = output_dir / "entities.parquet"
if entities_file.exists():
    df = pd.read_parquet(entities_file)
    print(f"实体数量: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    print("\n前10个实体:")
    if 'title' in df.columns:
        print(df[['title', 'type']].head(10))
    else:
        print(df.head(10))
else:
    print("entities.parquet 不存在")

community_file = output_dir / "community_reports.parquet"
if community_file.exists():
    df = pd.read_parquet(community_file)
    print(f"\n社区报告数量: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
