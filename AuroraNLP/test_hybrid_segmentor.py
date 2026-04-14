from AuroraNLP import HybridSegmentor, Dictionary, HybridConfig, HybridStrategy

# 测试混合分词器
def test_hybrid_segmentor():
    # 创建字典
    dictionary = Dictionary(load_default=True)
    
    # 创建混合分词器
    hybrid_segmentor = HybridSegmentor(dictionary=dictionary)
    
    # 测试文本
    test_text = "我爱自然语言处理"
    
    # 测试不同的融合策略
    strategies = [
        HybridStrategy.VOTE,
        HybridStrategy.WEIGHTED,
        HybridStrategy.CASCADE,
        HybridStrategy.ADAPTIVE,
        HybridStrategy.CONFIDENCE
    ]
    
    print("测试混合分词器...")
    print(f"测试文本: {test_text}")
    
    for strategy in strategies:
        # 设置策略
        config = HybridConfig(strategy=strategy)
        hybrid_segmentor.set_config(config)
        
        # 分词
        result = hybrid_segmentor.segment(test_text)
        
        # 分词带详情
        result_with_details, details = hybrid_segmentor.segment_with_details(test_text)
        
        print(f"\n策略: {strategy.value}")
        print(f"分词结果: {result}")
        print(f"带详情结果: {result_with_details}")
        print(f"最终置信度: {details['final_confidence']:.4f}")
        
        # 打印各个分词器的结果
        print("各分词器结果:")
        for seg_result in details['results']:
            print(f"  - {seg_result['segmenter']}: {seg_result['words']} (置信度: {seg_result['confidence']:.4f})")
    
    # 测试获取可用的分词器和策略
    print("\n可用的分词器:")
    print(hybrid_segmentor.get_available_segmenters())
    
    print("\n可用的融合策略:")
    print(hybrid_segmentor.get_available_strategies())
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_hybrid_segmentor()