#!/usr/bin/env python3
import litellm

try:
    response = litellm.embedding(
        model='ollama/nomic-embed-text',
        input=['hello world'],
        api_base='http://localhost:11434'
    )
    print('Success!')
    print(f'Embedding dimensions: {len(response.data[0]["embedding"])}')
except Exception as e:
    print(f'Error: {e}')
