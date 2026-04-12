import google.generativeai as genai
import os

# 设置API密钥
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    api_key = input("请输入您的Gemini API密钥: ")

genai.configure(api_key=api_key)

print("正在列出所有支持generateContent的Gemini模型...")
print("=" * 80)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"模型名称: {m.name}")
        print(f"支持的方法: {m.supported_generation_methods}")
        print(f"版本: {m.version}")
        print("-" * 80)

print("完成！")
