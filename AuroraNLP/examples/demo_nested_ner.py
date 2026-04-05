"""
实体嵌套识别示例

演示如何使用NestedNERRecognizer识别嵌套实体
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP import (
    NestedEntity,
    EntityHierarchy,
    NestedNERRecognizer,
    create_nested_ner_corpus,
)


def main():
    print("=" * 60)
    print("AuroraNLP 实体嵌套识别示例")
    print("=" * 60)
    
    recognizer = NestedNERRecognizer()
    
    print("\n1. 准备训练数据...")
    corpus = create_nested_ner_corpus()
    print(f"   训练语料数量: {len(corpus)}")
    
    for i, (text, entities) in enumerate(corpus[:3], 1):
        print(f"   示例 {i}: {text}")
        for entity in entities:
            print(f"      - {entity.text} ({entity.entity_type}) [{entity.start}:{entity.end}]")
    
    print("\n2. 训练嵌套实体识别模型...")
    recognizer.train(corpus, max_iter=50, verbose=True)
    print("   训练完成!")
    
    print("\n3. 测试嵌套实体识别...")
    test_texts = [
        "北京协和医院是一家著名医院",
        "上海交通大学医学院",
        "广东省深圳市南山区",
        "清华大学计算机系",
    ]
    
    for text in test_texts:
        print(f"\n   文本: {text}")
        
        hierarchy = recognizer.recognize_nested(text, max_levels=3)
        
        print(f"   实体数量: {len(hierarchy.all_entities)}")
        print(f"   最大深度: {hierarchy.get_max_depth()}")
        
        if hierarchy.all_entities:
            print("   实体树结构:")
            print(hierarchy.to_tree_string())
        
        annotated = recognizer.annotate_nested(text, max_levels=3)
        print(f"   标注结果: {annotated}")
    
    print("\n4. 测试实体层次结构操作...")
    
    hierarchy = recognizer.recognize_nested("广东省深圳市南山区", max_levels=3)
    
    if hierarchy.all_entities:
        print(f"\n   所有实体:")
        for entity in hierarchy.flatten():
            print(f"      - {entity.text} ({entity.entity_type}) level={entity.level}")
        
        print(f"\n   按类型查询 (LOC):")
        loc_entities = hierarchy.get_entities_by_type("LOC")
        for entity in loc_entities:
            print(f"      - {entity.text} level={entity.level}")
        
        print(f"\n   按层级查询 (level=0):")
        level_0_entities = hierarchy.get_entities_at_level(0)
        for entity in level_0_entities:
            print(f"      - {entity.text} ({entity.entity_type})")
        
        print(f"\n   嵌套实体:")
        nested_entities = hierarchy.get_nested_entities()
        for entity in nested_entities:
            print(f"      - {entity.text} ({entity.entity_type}) has {len(entity.children)} children")
    
    print("\n5. 测试嵌套规则...")
    print(f"\n   当前嵌套规则:")
    for parent_type, children_types in recognizer.nesting_rules.items():
        print(f"      {parent_type} -> {children_types}")
    
    print(f"\n   测试嵌套规则:")
    print(f"      ORG 可以包含 LOC: {recognizer.can_nest('ORG', 'LOC')}")
    print(f"      ORG 可以包含 PER: {recognizer.can_nest('ORG', 'PER')}")
    print(f"      LOC 可以包含 ORG: {recognizer.can_nest('LOC', 'ORG')}")
    print(f"      NUM 可以包含 ORG: {recognizer.can_nest('NUM', 'ORG')}")
    
    print(f"\n   修改嵌套规则:")
    recognizer.set_nesting_rule('ORG', ['LOC', 'PER', 'TIME'])
    print(f"      ORG -> {recognizer.nesting_rules['ORG']}")
    
    print("\n6. 测试模型保存和加载...")
    model_path = "nested_ner_model.pkl"
    
    recognizer.save_model(model_path)
    print(f"   模型已保存到: {model_path}")
    
    new_recognizer = NestedNERRecognizer()
    new_recognizer.load_model(model_path)
    print(f"   模型已加载")
    
    hierarchy = new_recognizer.recognize_nested("北京协和医院", max_levels=3)
    print(f"   验证加载的模型: 识别到 {len(hierarchy.all_entities)} 个实体")
    
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"   清理临时文件: {model_path}")
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
