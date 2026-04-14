from AuroraNLP import NetworkDictionary

# 测试 NetworkDictionary 类的基本功能
def test_network_dictionary():
    print("测试 NetworkDictionary 初始化...")
    try:
        # 初始化网络词典
        network_dict = NetworkDictionary(load_default=True)
        print("✓ 网络词典初始化成功")
        
        # 测试获取统计信息
        stats = network_dict.get_statistics()
        print(f"✓ 统计信息获取成功: 总词数={stats['total_words']}")
        
        # 测试获取最近的词
        recent_words = network_dict.get_recent_words(days=7)
        print(f"✓ 最近词获取成功: {len(recent_words)}个词")
        
        # 测试获取过期的词
        expired_words = network_dict.get_expired_words()
        print(f"✓ 过期词获取成功: {len(expired_words)}个词")
        
        print("\n所有测试通过！")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_network_dictionary()
