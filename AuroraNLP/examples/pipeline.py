#!/usr/bin/env python3
"""
Pipeline 系统示例
演示 AuroraNLP 的 Pipeline 功能
"""

from AuroraNLP import Pipeline, Segmentor, POSTagger, NERRecognizer


def basic_pipeline():
    """基础 Pipeline"""
    print("=" * 60)
    print("1. 基础 Pipeline")
    print("=" * 60)
    
    # 创建 Pipeline
    nlp = Pipeline()
    nlp.add_component(Segmentor())
    nlp.add_component(POSTagger())
    nlp.add_component(NERRecognizer())
    
    # 处理文本
    doc = nlp("张三在腾讯公司工作")
    
    print(f"原文: {doc.text}")
    print(f"分词: {' / '.join([token.text for token in doc.tokens])}")
    print(f"词性: {' / '.join([f'{token.text}({token.pos})' for token in doc.tokens])}")
    print(f"实体: {[f'{e.text}({e.type})' for e in doc.ents]}")
    print()


def doc_span_token():
    """Doc/Span/Token 使用"""
    print("=" * 60)
    print("2. Doc/Span/Token 使用")
    print("=" * 60)
    
    nlp = Pipeline()
    nlp.add_component(Segmentor())
    nlp.add_component(POSTagger())
    
    doc = nlp("我喜欢吃苹果和香蕉")
    
    print("Doc 对象:")
    print(f"  原文: {doc.text}")
    print(f"  词数: {len(doc.tokens)}")
    print()
    
    print("Token 对象:")
    for i, token in enumerate(doc.tokens):
        print(f"  Token {i}: {token.text} (POS: {token.pos}, 索引: [{token.start}-{token.end}])")
    print()
    
    print("Span 对象:")
    span = doc[2:5]  # "吃苹果和"
    print(f"  Span 文本: {span.text}")
    print(f"  Span token: {[token.text for token in span]}")
    print()


def custom_component():
    """自定义组件"""
    print("=" * 60)
    print("3. 自定义组件")
    print("=" * 60)
    
    from AuroraNLP import PipelineComponent
    
    class WordCounter(PipelineComponent):
        name = "word_counter"
        
        def __call__(self, doc):
            doc.word_count = len(doc.tokens)
            return doc
    
    class CharacterCounter(PipelineComponent):
        name = "char_counter"
        
        def __call__(self, doc):
            doc.char_count = len(doc.text)
            return doc
    
    nlp = Pipeline()
    nlp.add_component(Segmentor())
    nlp.add_component(WordCounter())
    nlp.add_component(CharacterCounter())
    
    doc = nlp("这是一段测试文本")
    
    print(f"原文: {doc.text}")
    print(f"词数: {doc.word_count}")
    print(f"字符数: {doc.char_count}")
    print()


def pipeline_config():
    """Pipeline 配置"""
    print("=" * 60)
    print("4. Pipeline 配置")
    print("=" * 60)
    
    from AuroraNLP import PipelineConfig
    
    config = PipelineConfig({
        "components": [
            {"name": "segmentor", "enabled": True},
            {"name": "pos_tagger", "enabled": True},
            {"name": "ner", "enabled": True},
        ]
    })
    
    nlp = Pipeline(config=config)
    nlp.add_component(Segmentor(), name="segmentor")
    nlp.add_component(POSTagger(), name="pos_tagger")
    nlp.add_component(NERRecognizer(), name="ner")
    
    doc = nlp("李四在北京百度工作")
    
    print(f"原文: {doc.text}")
    print(f"分词: {' / '.join([token.text for token in doc.tokens])}")
    print()


def batch_processing():
    """批量处理"""
    print("=" * 60)
    print("5. 批量处理")
    print("=" * 60)
    
    nlp = Pipeline()
    nlp.add_component(Segmentor())
    
    texts = [
        "第一句话",
        "第二句话",
        "第三句话",
        "第四句话",
        "第五句话",
    ]
    
    docs = nlp.pipe(texts)
    
    print(f"处理了 {len(docs)} 个文档:")
    for i, doc in enumerate(docs):
        print(f"  文档 {i+1}: {' / '.join([token.text for token in doc.tokens])}")
    print()


def main():
    print("\n")
    basic_pipeline()
    doc_span_token()
    custom_component()
    pipeline_config()
    batch_processing()
    print("\n")


if __name__ == "__main__":
    main()
