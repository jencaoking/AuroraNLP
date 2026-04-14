import os
import json
import re
from typing import List, Dict, Optional, Tuple, Union


class CorpusBuilder:
    """训练语料库构建器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化语料库构建器
        
        Args:
            data_dir: 数据目录路径，默认为包内的 data 目录
        """
        if data_dir is None:
            # 使用包内的 data 目录
            import os
            self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        else:
            self.data_dir = data_dir
        self.corpus_types = {
            "people_daily": "人民日报语料",
            "ctb": "CTB语料",
            "msra": "MSRA语料",
            "custom": "自有语料"
        }
    
    def load_corpus(self, corpus_type: str, file_path: str) -> List[str]:
        """
        加载语料库文件
        
        Args:
            corpus_type: 语料类型
            file_path: 文件路径
            
        Returns:
            语料文本列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"语料文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        return lines
    
    def preprocess_corpus(self, lines: List[str], corpus_type: str) -> List[str]:
        """
        预处理语料
        
        Args:
            lines: 原始语料行
            corpus_type: 语料类型
            
        Returns:
            预处理后的语料行
        """
        preprocessed_lines = []
        
        for line in lines:
            # 通用预处理
            line = line.strip()
            if not line:
                continue
            
            # 根据语料类型进行特定处理
            if corpus_type == "people_daily":
                # 人民日报语料处理
                line = self._process_people_daily(line)
            elif corpus_type == "ctb":
                # CTB语料处理
                line = self._process_ctb(line)
            elif corpus_type == "msra":
                # MSRA语料处理
                line = self._process_msra(line)
            
            if line:
                preprocessed_lines.append(line)
        
        return preprocessed_lines
    
    def _process_people_daily(self, line: str) -> str:
        """处理人民日报语料"""
        # 移除标记和注释
        line = re.sub(r'\\[.*?\\]', '', line)
        line = re.sub(r'\\(.*?\\)', '', line)
        return line
    
    def _process_ctb(self, line: str) -> str:
        """处理CTB语料"""
        # 移除XML标记
        line = re.sub(r'<.*?>', '', line)
        return line
    
    def _process_msra(self, line: str) -> str:
        """处理MSRA语料"""
        # MSRA语料通常是分词标注格式，保留原始格式
        return line
    
    def convert_to_standard_format(self, lines: List[str], format_type: str = "segmented") -> List[str]:
        """
        转换为标准格式
        
        Args:
            lines: 预处理后的语料行
            format_type: 目标格式类型
                - "segmented": 分词格式 (词1 词2 词3)
                - "tagged": 词性标注格式 (词1/词性 词2/词性)
                - "plain": 纯文本格式
                
        Returns:
            转换后的语料行
        """
        converted_lines = []
        
        for line in lines:
            if format_type == "segmented":
                # 简单分词处理（实际应用中可能需要更复杂的分词）
                # 这里假设输入已经是分词格式
                converted_lines.append(line)
            elif format_type == "tagged":
                # 简单词性标注处理
                # 这里假设输入已经是标注格式
                converted_lines.append(line)
            elif format_type == "plain":
                # 转换为纯文本
                # 移除分词标记
                plain_line = re.sub(r'\\s+', '', line)
                converted_lines.append(plain_line)
        
        return converted_lines
    
    def build_corpus(self, corpus_config: Dict[str, Union[str, List[str]]]) -> str:
        """
        构建综合语料库
        
        Args:
            corpus_config: 语料配置
                {
                    "output_path": "输出文件路径",
                    "corpora": [
                        {"type": "people_daily", "path": "文件路径"},
                        {"type": "ctb", "path": "文件路径"},
                        {"type": "msra", "path": "文件路径"},
                        {"type": "custom", "path": "文件路径"}
                    ],
                    "format": "segmented"  # 目标格式
                }
                
        Returns:
            构建完成的语料库文件路径
        """
        output_path = corpus_config.get("output_path", os.path.join(self.data_dir, "train_corpus.txt"))
        corpora = corpus_config.get("corpora", [])
        target_format = corpus_config.get("format", "segmented")
        
        all_lines = []
        
        for corpus_info in corpora:
            corpus_type = corpus_info.get("type")
            corpus_path = corpus_info.get("path")
            
            if not corpus_type or not corpus_path:
                continue
            
            try:
                # 加载语料
                lines = self.load_corpus(corpus_type, corpus_path)
                # 预处理
                preprocessed_lines = self.preprocess_corpus(lines, corpus_type)
                # 转换格式
                converted_lines = self.convert_to_standard_format(preprocessed_lines, target_format)
                # 添加到总语料
                all_lines.extend(converted_lines)
                
                print(f"成功加载 {self.corpus_types.get(corpus_type, corpus_type)}: {len(converted_lines)} 条")
            except Exception as e:
                print(f"加载 {self.corpus_types.get(corpus_type, corpus_type)} 失败: {e}")
        
        # 保存构建的语料库
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_lines))
        
        print(f"语料库构建完成，保存到: {output_path}")
        print(f"总语料条数: {len(all_lines)}")
        
        return output_path
    
    def split_corpus(self, corpus_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[str, str, str]:
        """
        分割语料库为训练集、验证集和测试集
        
        Args:
            corpus_path: 语料库文件路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            (训练集路径, 验证集路径, 测试集路径)
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"语料库文件不存在: {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        total = len(lines)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_lines = lines[:train_size]
        val_lines = lines[train_size:train_size + val_size]
        test_lines = lines[train_size + val_size:]
        
        # 生成输出路径
        base_dir = os.path.dirname(corpus_path)
        base_name = os.path.splitext(os.path.basename(corpus_path))[0]
        
        train_path = os.path.join(base_dir, f"{base_name}_train.txt")
        val_path = os.path.join(base_dir, f"{base_name}_val.txt")
        test_path = os.path.join(base_dir, f"{base_name}_test.txt")
        
        # 保存分割后的文件
        for path, data in [(train_path, train_lines), (val_path, val_lines), (test_path, test_lines)]:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(data))
            print(f"保存 {os.path.basename(path)}: {len(data)} 条")
        
        return train_path, val_path, test_path
    
    def statistics(self, corpus_path: str) -> Dict[str, Union[int, float]]:
        """
        计算语料库统计信息
        
        Args:
            corpus_path: 语料库文件路径
            
        Returns:
            统计信息字典
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"语料库文件不存在: {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        total_lines = len(lines)
        total_tokens = 0
        total_chars = 0
        
        for line in lines:
            # 计算词数（假设是分词格式）
            tokens = line.split()
            total_tokens += len(tokens)
            # 计算字符数（去除空格）
            total_chars += len(re.sub(r'\\s+', '', line))
        
        avg_tokens_per_line = total_tokens / total_lines if total_lines > 0 else 0
        avg_chars_per_line = total_chars / total_lines if total_lines > 0 else 0
        
        return {
            "total_lines": total_lines,
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "avg_tokens_per_line": round(avg_tokens_per_line, 2),
            "avg_chars_per_line": round(avg_chars_per_line, 2)
        }


class CorpusManager:
    """语料库管理器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化语料库管理器
        
        Args:
            data_dir: 数据目录路径，默认为包内的 data 目录
        """
        import os
        if data_dir is None:
            # 使用包内的 data 目录
            self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        else:
            self.data_dir = data_dir
        self.builder = CorpusBuilder(self.data_dir)
        self.corpus_registry = os.path.join(self.data_dir, "corpus_registry.json")
        self._load_registry()
    
    def _load_registry(self):
        """加载语料库注册表"""
        if os.path.exists(self.corpus_registry):
            with open(self.corpus_registry, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"corpora": []}
    
    def _save_registry(self):
        """保存语料库注册表"""
        os.makedirs(os.path.dirname(self.corpus_registry), exist_ok=True)
        with open(self.corpus_registry, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
    
    def register_corpus(self, name: str, corpus_type: str, path: str, description: str = ""):
        """
        注册语料库
        
        Args:
            name: 语料库名称
            corpus_type: 语料类型
            path: 语料文件路径
            description: 语料描述
        """
        from datetime import datetime
        corpus_info = {
            "name": name,
            "type": corpus_type,
            "path": path,
            "description": description,
            "registered_at": datetime.now().isoformat()
        }
        
        # 检查是否已存在
        existing = next((c for c in self.registry["corpora"] if c["name"] == name), None)
        if existing:
            existing.update(corpus_info)
        else:
            self.registry["corpora"].append(corpus_info)
        
        self._save_registry()
        print(f"语料库 {name} 注册成功")
    
    def list_corpora(self) -> List[Dict[str, str]]:
        """
        列出所有注册的语料库
        
        Returns:
            语料库信息列表
        """
        return self.registry["corpora"]
    
    def build_combined_corpus(self, name: str, corpus_names: List[str], output_format: str = "segmented") -> str:
        """
        构建组合语料库
        
        Args:
            name: 组合语料库名称
            corpus_names: 要组合的语料库名称列表
            output_format: 输出格式
            
        Returns:
            组合语料库文件路径
        """
        corpora = []
        
        for corpus_name in corpus_names:
            corpus = next((c for c in self.registry["corpora"] if c["name"] == corpus_name), None)
            if corpus:
                corpora.append({"type": corpus["type"], "path": corpus["path"]})
            else:
                print(f"警告: 语料库 {corpus_name} 未注册")
        
        if not corpora:
            raise ValueError("没有找到有效的语料库")
        
        output_path = os.path.join(self.data_dir, f"{name}_corpus.txt")
        
        config = {
            "output_path": output_path,
            "corpora": corpora,
            "format": output_format
        }
        
        result_path = self.builder.build_corpus(config)
        
        # 注册组合语料库
        self.register_corpus(
            name=name,
            corpus_type="combined",
            path=result_path,
            description=f"组合语料库: {', '.join(corpus_names)}"
        )
        
        return result_path
    
    def get_corpus(self, name: str) -> Optional[Dict[str, str]]:
        """
        获取语料库信息
        
        Args:
            name: 语料库名称
            
        Returns:
            语料库信息字典
        """
        return next((c for c in self.registry["corpora"] if c["name"] == name), None)
    
    def remove_corpus(self, name: str):
        """
        移除语料库注册
        
        Args:
            name: 语料库名称
        """
        self.registry["corpora"] = [c for c in self.registry["corpora"] if c["name"] != name]
        self._save_registry()
        print(f"语料库 {name} 已移除")
