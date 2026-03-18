"""检查LanceDB向量索引的维度"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb

db = lancedb.connect("data/output/output/lancedb")
print(f"Tables: {db.table_names()}")

for table_name in db.table_names():
    table = db.open_table(table_name)
    df = table.to_pandas()
    print(f"\n=== {table_name} ===")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Rows: {len(df)}")
    
    # 检查向量维度
    for col in df.columns:
        if 'vector' in col.lower() or 'embedding' in col.lower():
            sample = df[col].iloc[0]
            if hasattr(sample, '__len__'):
                print(f"  {col}: dimension = {len(sample)}")
            break
