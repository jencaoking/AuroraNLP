"""
CRF (Conditional Random Field) 条件随机场示例

本示例展示如何使用AuroraNLP中的CRF功能进行中文分词
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP import Segmentor

def main():
    print("=" * 60)
    print("CRF 条件随机场分词示例")
    print("=" * 60)
    
    print("\n1. 创建分词器")
    segmentor = Segmentor()
    
    print("\n2. 准备训练语料")
    corpus = [
        ['我', '爱', '中国'],
        ['他', '是', '学生'],
        ['北京', '是', '首都'],
        ['我', '喜欢', '学习'],
        ['自然语言', '处理', '很', '有趣'],
        ['深度', '学习', '改变', '世界'],
        ['机器', '学习', '是', '人工智能', '的', '基础'],
        ['中文', '分词', '是', 'NLP', '的', '基础', '任务'],
    ]
    
    print(f"训练语料包含 {len(corpus)} 个句子")
    for i, sentence in enumerate(corpus[:3], 1):
        print(f"  {i}. {' / '.join(sentence)}")
    
    print("\n3. 训练CRF模型")
    print("训练参数:")
    print("  - 学习率: 0.1")
    print("  - L2正则化: 0.01")
    print("  - 最大迭代次数: 50")
    
    segmentor.train_crf(
        corpus,
        learning_rate=0.1,
        l2_reg=0.01,
        max_iter=50,
        verbose=True
    )
    
    print("\n4. 查看模型信息")
    model_info = segmentor.get_crf_model_info()
    print(f"模型状态: {'已训练' if model_info['trained'] else '未训练'}")
    print(f"标签数量: {model_info['num_tags']}")
    print(f"特征数量: {model_info['num_features']}")
    print(f"特征模板数量: {model_info['num_feature_templates']}")
    
    print("\n5. 使用CRF模式进行分词")
    test_texts = [
        "我爱中国",
        "他是学生",
        "自然语言处理很有趣",
        "深度学习改变世界",
        "中文分词很重要"
    ]
    
    for text in test_texts:
        words = segmentor.segment(text, mode='crf')
        print(f"原文: {text}")
        print(f"分词: {' / '.join(words)}")
        print()
    
    print("\n6. 查看分词状态序列")
    text = "我爱中国"
    states = segmentor.segment_with_crf_states(text)
    print(f"文本: {text}")
    print("状态序列:")
    for char, state in states:
        state_name = {
            'B': '词首',
            'M': '词中',
            'E': '词尾',
            'S': '单字词'
        }.get(state, state)
        print(f"  {char} -> {state} ({state_name})")
    
    print("\n7. 保存和加载模型")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        model_path = f.name
    
    try:
        segmentor.save_crf_model(model_path)
        print(f"模型已保存到: {model_path}")
        
        new_segmentor = Segmentor()
        new_segmentor.load_crf_model(model_path)
        print("模型已重新加载")
        
        text = "我爱学习"
        words1 = segmentor.segment(text, mode='crf')
        words2 = new_segmentor.segment(text, mode='crf')
        
        print(f"\n原始模型分词: {' / '.join(words1)}")
        print(f"加载模型分词: {' / '.join(words2)}")
        print(f"结果一致: {words1 == words2}")
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"\n临时文件已删除: {model_path}")
    
    print("\n8. 对比不同分词模式")
    text = "我爱自然语言处理"
    
    print(f"原文: {text}")
    print(f"双向最大匹配: {' / '.join(segmentor.segment(text, mode='bidirectional'))}")
    print(f"CRF分词: {' / '.join(segmentor.segment(text, mode='crf'))}")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
