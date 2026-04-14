import json
import os
import random
from typing import List, Dict, Optional, Tuple, Union, Set
from collections import defaultdict
import numpy as np

class CorpusAnnotator:
    def __init__(self, corpus_path: str = None):
        self.corpus_path = corpus_path
        self.annotated_data = []
        self.unannotated_data = []
        self.quality_metrics = defaultdict(list)
        
        if corpus_path and os.path.exists(corpus_path):
            self.load_corpus(corpus_path)
    
    def load_corpus(self, corpus_path: str):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.unannotated_data.append(line)
    
    def save_annotated_corpus(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.annotated_data:
                if isinstance(item, dict):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    f.write(item + '\n')
    
    def semi_supervised_annotate(self, model, sample_size: int = 100, confidence_threshold: float = 0.8):
        if not self.unannotated_data:
            return []
        
        samples = random.sample(self.unannotated_data, min(sample_size, len(self.unannotated_data)))
        annotated_samples = []
        
        for sample in samples:
            try:
                result = model.analyze(sample)
                if hasattr(result, 'confidence') and result.confidence >= confidence_threshold:
                    annotated_samples.append({
                        'text': sample,
                        'annotation': result.to_dict(),
                        'confidence': result.confidence,
                        'method': 'semi_supervised'
                    })
                    self.annotated_data.append({
                        'text': sample,
                        'annotation': result.to_dict(),
                        'confidence': result.confidence,
                        'method': 'semi_supervised'
                    })
                    self.unannotated_data.remove(sample)
            except Exception as e:
                continue
        
        return annotated_samples
    
    def active_learning_sample(self, model, sample_size: int = 50, strategy: str = 'uncertainty'):
        if not self.unannotated_data:
            return []
        
        scores = []
        for i, sample in enumerate(self.unannotated_data):
            try:
                if strategy == 'uncertainty':
                    result = model.analyze(sample)
                    if hasattr(result, 'confidence'):
                        uncertainty = 1 - result.confidence
                        scores.append((i, uncertainty))
                elif strategy == 'diversity':
                    diversity_score = self._calculate_diversity(sample)
                    scores.append((i, diversity_score))
            except Exception as e:
                scores.append((i, 0.0))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in scores[:sample_size]]
        selected_samples = [self.unannotated_data[idx] for idx in selected_indices]
        
        return selected_samples
    
    def _calculate_diversity(self, sample: str) -> float:
        chars = set(sample)
        return len(chars) / len(sample) if sample else 0
    
    def evaluate_annotation_quality(self, annotated_data: List[Dict] = None):
        if not annotated_data:
            annotated_data = self.annotated_data
        
        if not annotated_data:
            return {}
        
        metrics = {
            'average_confidence': 0.0,
            'confidence_std': 0.0,
            'annotation_coverage': 0.0,
            'method_distribution': defaultdict(int)
        }
        
        confidences = []
        for item in annotated_data:
            if isinstance(item, dict) and 'confidence' in item:
                confidences.append(item['confidence'])
            if isinstance(item, dict) and 'method' in item:
                metrics['method_distribution'][item['method']] += 1
        
        if confidences:
            metrics['average_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
        
        metrics['annotation_coverage'] = len(annotated_data) / (len(annotated_data) + len(self.unannotated_data)) if (len(annotated_data) + len(self.unannotated_data)) > 0 else 0
        
        for key, value in metrics.items():
            self.quality_metrics[key].append(value)
        
        return metrics
    
    def get_quality_report(self):
        report = {}
        for key, values in self.quality_metrics.items():
            if values:
                report[f'{key}_mean'] = np.mean(values)
                report[f'{key}_std'] = np.std(values)
                report[f'{key}_latest'] = values[-1]
        return report
    
    def add_manual_annotation(self, text: str, annotation: Dict, confidence: float = 1.0):
        self.annotated_data.append({
            'text': text,
            'annotation': annotation,
            'confidence': confidence,
            'method': 'manual'
        })
        if text in self.unannotated_data:
            self.unannotated_data.remove(text)
    
    def get_statistics(self):
        return {
            'total_samples': len(self.annotated_data) + len(self.unannotated_data),
            'annotated_samples': len(self.annotated_data),
            'unannotated_samples': len(self.unannotated_data),
            'annotation_rate': len(self.annotated_data) / (len(self.annotated_data) + len(self.unannotated_data)) if (len(self.annotated_data) + len(self.unannotated_data)) > 0 else 0
        }

class AnnotationManager:
    def __init__(self):
        self.annotators = {}
        self.global_metrics = defaultdict(list)
    
    def create_annotator(self, name: str, corpus_path: str = None) -> CorpusAnnotator:
        annotator = CorpusAnnotator(corpus_path)
        self.annotators[name] = annotator
        return annotator
    
    def get_annotator(self, name: str) -> Optional[CorpusAnnotator]:
        return self.annotators.get(name)
    
    def evaluate_all_annotators(self):
        reports = {}
        for name, annotator in self.annotators.items():
            report = annotator.evaluate_annotation_quality()
            reports[name] = report
            for key, value in report.items():
                self.global_metrics[key].append(value)
        return reports
    
    def get_global_report(self):
        report = {}
        for key, values in self.global_metrics.items():
            if values:
                report[f'{key}_mean'] = np.mean(values)
                report[f'{key}_std'] = np.std(values)
        return report
    
    def save_all_annotations(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for name, annotator in self.annotators.items():
            output_path = os.path.join(output_dir, f'{name}_annotated.jsonl')
            annotator.save_annotated_corpus(output_path)

class ActiveLearningStrategy:
    @staticmethod
    def uncertainty_sampling(model, samples: List[str], n: int = 10):
        scores = []
        for sample in samples:
            try:
                result = model.analyze(sample)
                if hasattr(result, 'confidence'):
                    uncertainty = 1 - result.confidence
                    scores.append((sample, uncertainty))
            except Exception as e:
                scores.append((sample, 0.0))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [sample for sample, _ in scores[:n]]
    
    @staticmethod
    def diversity_sampling(samples: List[str], n: int = 10):
        scores = []
        for sample in samples:
            diversity = len(set(sample)) / len(sample) if sample else 0
            scores.append((sample, diversity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [sample for sample, _ in scores[:n]]
    
    @staticmethod
    def combination_sampling(model, samples: List[str], n: int = 10, alpha: float = 0.5):
        scores = []
        for sample in samples:
            try:
                result = model.analyze(sample)
                if hasattr(result, 'confidence'):
                    uncertainty = 1 - result.confidence
                    diversity = len(set(sample)) / len(sample) if sample else 0
                    combined = alpha * uncertainty + (1 - alpha) * diversity
                    scores.append((sample, combined))
            except Exception as e:
                scores.append((sample, 0.0))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [sample for sample, _ in scores[:n]]

class AnnotationQualityEvaluator:
    @staticmethod
    def calculate_precision(predicted: List[Dict], ground_truth: List[Dict]) -> float:
        if not predicted:
            return 0.0
        
        correct = 0
        for pred, truth in zip(predicted, ground_truth):
            if pred == truth:
                correct += 1
        
        return correct / len(predicted)
    
    @staticmethod
    def calculate_recall(predicted: List[Dict], ground_truth: List[Dict]) -> float:
        if not ground_truth:
            return 0.0
        
        correct = 0
        for pred, truth in zip(predicted, ground_truth):
            if pred == truth:
                correct += 1
        
        return correct / len(ground_truth)
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def evaluate_annotations(annotated_data: List[Dict], ground_truth: List[Dict]) -> Dict:
        if len(annotated_data) != len(ground_truth):
            raise ValueError("Annotated data and ground truth must have the same length")
        
        predicted = [item['annotation'] for item in annotated_data if isinstance(item, dict) and 'annotation' in item]
        truth = ground_truth
        
        precision = AnnotationQualityEvaluator.calculate_precision(predicted, truth)
        recall = AnnotationQualityEvaluator.calculate_recall(predicted, truth)
        f1 = AnnotationQualityEvaluator.calculate_f1_score(precision, recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }