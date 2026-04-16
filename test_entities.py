#!/usr/bin/env python3
import pandas as pd

# 加载实体
df = pd.read_parquet('data/graphrag_output/output/entities.parquet')

print(f'总实体数: {len(df)}')
print(f'列名: {list(df.columns)}')

# 搜索包含'深圳'的实体
shenzhen_entities = df[df['title'].str.contains('深圳', case=False, na=False) | 
                        df['description'].str.contains('深圳', case=False, na=False)]

print(f'\n包含深圳的实体: {len(shenzhen_entities)} 个')
print('\n前10个:')
for _, row in shenzhen_entities.head(10).iterrows():
    print(f"- {row['title']} ({row['type']}): {row['description'][:100]}...")

# 搜索包含'1990'的实体
entities_1990 = df[df['title'].str.contains('1990', case=False, na=False) | 
                   df['description'].str.contains('1990', case=False, na=False)]

print(f'\n包含1990的实体: {len(entities_1990)} 个')
