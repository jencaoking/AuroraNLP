import sys
import os

print("Python path:")
for path in sys.path:
    print(f"  {path}")

# 检查ambiguity模块的导入
print("\nImporting AmbiguityResult from AuroraNLP.ambiguity:")
try:
    from AuroraNLP.ambiguity import AmbiguityResult
    print(f"  Successfully imported from: {AmbiguityResult.__module__}")
    print(f"  Class location: {AmbiguityResult.__class__}")
except Exception as e:
    print(f"  Error: {e}")

# 检查Segmentor的导入
print("\nImporting Segmentor:")
try:
    from AuroraNLP import Segmentor
    print(f"  Successfully imported from: {Segmentor.__module__}")
    
    # 创建Segmentor实例并测试detect_ambiguity方法
    seg = Segmentor(load_default_dict=False)
    seg.add_word("研究", "v")
    seg.add_word("研究生", "n")
    
    result = seg.detect_ambiguity("研究生")
    print(f"  detect_ambiguity result: {result}")
    print(f"  Result type: {type(result)}")
    print(f"  Result class: {result.__class__}")
    print(f"  Result module: {result.__class__.__module__}")
    
    # 检查类型是否匹配
    from AuroraNLP.ambiguity import AmbiguityResult as ImportedAmbiguityResult
    print(f"  Is instance: {isinstance(result, ImportedAmbiguityResult)}")
    print(f"  Same class: {result.__class__ is ImportedAmbiguityResult}")
    print(f"  Imported class module: {ImportedAmbiguityResult.__module__}")
    
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
