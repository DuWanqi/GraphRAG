#!/usr/bin/env python3
"""
处理历史事件txt文件脚本
将隔一行一个事件的txt文件转换为系统可索引的格式
只使用 --- 作为分隔符，保留原始内容
"""

from pathlib import Path


def process_txt_file(input_path: str, output_path: str):
    """
    处理txt文件，将隔一行一个事件的格式转换为用---分隔的格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    print(f"处理文件：{input_path}")
    
    # 读取所有行
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 收集所有非空行（每个非空行就是一个事件）
    events = []
    for line in lines:
        line = line.strip()
        if line:  # 只保留非空行
            events.append(line)
    
    print(f"  找到 {len(events)} 个事件")
    
    # 写入输出文件，用---分隔
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, event in enumerate(events):
            if i > 0:
                f.write('\n---\n\n')
            f.write(event)
            f.write('\n')
    
    print(f"  输出到：{output_path}")
    print(f"  完成！\n")


def main():
    """主函数"""
    input_dir = Path(r'd:\projects\Capstone\GraphRAG\data\output\input')
    output_dir = Path(r'd:\projects\Capstone\GraphRAG\data\input')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理深圳文件
    shenzhen_input = input_dir / 'shenzhen.txt'
    shenzhen_output = output_dir / 'shenzhen_processed.txt'
    if shenzhen_input.exists():
        process_txt_file(shenzhen_input, shenzhen_output)
    
    # 处理广东文件
    guangdong_input = input_dir / 'guangdong.txt'
    guangdong_output = output_dir / 'guangdong_processed.txt'
    if guangdong_input.exists():
        process_txt_file(guangdong_input, guangdong_output)
    
    # 处理中国文件
    zhongguo_input = input_dir / 'zhongguo.txt'
    zhongguo_output = output_dir / 'zhongguo_processed.txt'
    if zhongguo_input.exists():
        process_txt_file(zhongguo_input, zhongguo_output)
    
    print("所有文件处理完成！")
    print(f"处理后的文件保存在：{output_dir}")


if __name__ == "__main__":
    main()
