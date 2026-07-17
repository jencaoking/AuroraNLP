#!/usr/bin/env python3
"""
命名实体识别示例
演示 AuroraNLP 的 NER 功能
"""

from AuroraNLP import NERRecognizer


def basic_ner():
    """基础 NER"""
    print("=" * 60)
    print("1. 基础命名实体识别")
    print("=" * 60)
    
    ner = NERRecognizer()
    text = "张三在腾讯公司位于深圳南山区"
    entities = ner.recognize(text)
    
    print(f"原文: {text}")
    print("识别到的实体:")
    for entity in entities:
        print(f"  {entity.text} ({entity.type} [{entity.start}-{entity.end}")
    print()


def nested_ner():
    """嵌套 NER"""
    print("=" * 60)
    print("2. 嵌套命名实体识别")
    print("=" * 60)
    
    from AuroraNLP import NestedNERRecognizer
    
    nested_ner = NestedNERRecognizer()
    text = "中华人民共和国北京市朝阳区"
    entities = nested_ner.recognize(text)
    
    print(f"原文: {text}")
    print("识别到的实体（嵌套）:")
    for entity in entities:
        print(f"  {entity.text} ({entity.type} [{entity.start}-{entity.end}]")
    print()


def more_examples():
    """更多示例"""
    print("=" * 60)
    print("3. 更多示例")
    print("=" * 60)
    
    ner = NERRecognizer()
    
    examples = [
        "李四在北京阿里巴巴工作",
        "2023年10月1日在上海举办了会议",
        "王五博士毕业于清华大学",
        "赵六在招商银行办理了信用卡",
    ]
    
    for text in examples:
        entities = ner.recognize(text)
        print(f"原文: {text}")
        for entity in entities:
            print(f"  {entity.text} ({entity.type}")
        print()


def main():
    print("\n")
    basic_ner()
    nested_ner()
    more_examples()
    print("\n")


if __name__ == "__main__":
    main()
