from typing import List, Tuple, Optional, Dict, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

if TYPE_CHECKING:
    from .dictionary import Dictionary
    from .lattice import Lattice, LatticeEdge


class AmbiguityType(Enum):
    CROSS = "cross"
    COMBINATION = "combination"
    OVERLAP = "overlap"


@dataclass
class AmbiguityRegion:
    start: int
    end: int
    text: str
    ambiguity_type: AmbiguityType
    segmentations: List[List[str]] = field(default_factory=list)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)
    confidence: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"AmbiguityRegion('{self.text}', [{self.start}:{self.end}], "
            f"type={self.ambiguity_type.value}, segs={self.segmentations})"
        )
    
    def to_dict(self) -> Dict:
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'type': self.ambiguity_type.value,
            'segmentations': self.segmentations,
            'edges': self.edges,
            'confidence': self.confidence
        }


@dataclass
class AmbiguityResult:
    text: str
    total_ambiguities: int
    cross_count: int
    combination_count: int
    overlap_count: int
    regions: List[AmbiguityRegion] = field(default_factory=list)
    
    def has_ambiguity(self) -> bool:
        return self.total_ambiguities > 0
    
    def get_regions_by_type(self, ambiguity_type: AmbiguityType) -> List[AmbiguityRegion]:
        return [r for r in self.regions if r.ambiguity_type == ambiguity_type]
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'total_ambiguities': self.total_ambiguities,
            'cross_count': self.cross_count,
            'combination_count': self.combination_count,
            'overlap_count': self.overlap_count,
            'regions': [r.to_dict() for r in self.regions]
        }


class AmbiguityDetector:
    def __init__(self, dictionary: 'Dictionary', max_word_len: int = 15):
        self.dictionary = dictionary
        self.max_word_len = max_word_len
    
    def detect(self, text: str) -> AmbiguityResult:
        all_matches = self._get_all_word_matches(text)
        
        regions = []
        cross_regions = self._detect_cross_ambiguity(text, all_matches)
        combination_regions = self._detect_combination_ambiguity(text, all_matches)
        overlap_regions = self._detect_overlap_ambiguity(text, all_matches)
        
        regions.extend(cross_regions)
        regions.extend(combination_regions)
        regions.extend(overlap_regions)
        
        regions = self._merge_overlapping_regions(regions)
        
        cross_count = sum(1 for r in regions if r.ambiguity_type == AmbiguityType.CROSS)
        combination_count = sum(1 for r in regions if r.ambiguity_type == AmbiguityType.COMBINATION)
        overlap_count = sum(1 for r in regions if r.ambiguity_type == AmbiguityType.OVERLAP)
        
        return AmbiguityResult(
            text=text,
            total_ambiguities=len(regions),
            cross_count=cross_count,
            combination_count=combination_count,
            overlap_count=overlap_count,
            regions=regions
        )
    
    def _get_all_word_matches(self, text: str) -> Dict[int, List[Tuple[int, str]]]:
        matches: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        text_len = len(text)
        
        for start in range(text_len):
            max_len = min(self.max_word_len, text_len - start)
            for length in range(1, max_len + 1):
                word = text[start:start + length]
                if self.dictionary.search_in_dict(word):
                    matches[start].append((start + length, word))
        
        return matches
    
    def _detect_cross_ambiguity(self, text: str, matches: Dict[int, List[Tuple[int, str]]]) -> List[AmbiguityRegion]:
        regions = []
        text_len = len(text)
        processed = set()
        
        for start in range(text_len):
            if start not in matches or start in processed:
                continue
            
            edges = matches[start]
            if len(edges) < 2:
                continue
            
            cross_edges = []
            for i, (end1, word1) in enumerate(edges):
                for end2, word2 in edges[i + 1:]:
                    if end1 != end2:
                        cross_edges.append((start, end1, word1))
                        cross_edges.append((start, end2, word2))
            
            if not cross_edges:
                continue
            
            cross_edges = list(set(cross_edges))
            
            max_end = max(e[1] for e in cross_edges)
            region_text = text[start:max_end]
            
            segmentations = self._generate_cross_segmentations(start, max_end, matches, text)
            
            if len(segmentations) > 1:
                region = AmbiguityRegion(
                    start=start,
                    end=max_end,
                    text=region_text,
                    ambiguity_type=AmbiguityType.CROSS,
                    segmentations=segmentations,
                    edges=cross_edges,
                    confidence=self._calculate_confidence(segmentations)
                )
                regions.append(region)
                for i in range(start, max_end):
                    processed.add(i)
        
        return regions
    
    def _detect_combination_ambiguity(self, text: str, matches: Dict[int, List[Tuple[int, str]]]) -> List[AmbiguityRegion]:
        regions = []
        text_len = len(text)
        processed = set()
        
        for start in range(text_len):
            if start in processed:
                continue
            
            if start not in matches:
                continue
            
            edges = matches[start]
            if len(edges) < 2:
                continue
            
            has_cross = False
            for i, (end1, word1) in enumerate(edges):
                for end2, word2 in edges[i + 1:]:
                    if end1 != end2:
                        has_cross = True
                        break
                if has_cross:
                    break
            
            if has_cross:
                continue
            
            max_end = max(e[0] for e in edges)
            region_text = text[start:max_end]
            
            segmentations = []
            for end, word in edges:
                segmentations.append([word])
            
            if len(segmentations) > 1:
                region = AmbiguityRegion(
                    start=start,
                    end=max_end,
                    text=region_text,
                    ambiguity_type=AmbiguityType.COMBINATION,
                    segmentations=segmentations,
                    edges=[(start, end, word) for end, word in edges],
                    confidence=self._calculate_confidence(segmentations)
                )
                regions.append(region)
                processed.add(start)
        
        return regions
    
    def _detect_overlap_ambiguity(self, text: str, matches: Dict[int, List[Tuple[int, str]]]) -> List[AmbiguityRegion]:
        regions = []
        text_len = len(text)
        processed = set()
        
        for start in range(text_len):
            if start in processed or start not in matches:
                continue
            
            edges = matches[start]
            
            for end, word in edges:
                for next_start in range(start + 1, end):
                    if next_start in matches:
                        next_edges = matches[next_start]
                        for next_end, next_word in next_edges:
                            if next_end > end:
                                region_text = text[start:next_end]
                                
                                seg1 = self._build_segmentation(start, end, text, matches)
                                seg2 = self._build_segmentation(next_start, next_end, text, matches)
                                
                                if seg1 and seg2:
                                    segmentations = [seg1, seg2]
                                    edges_list = [
                                        (start, end, word),
                                        (next_start, next_end, next_word)
                                    ]
                                    
                                    region = AmbiguityRegion(
                                        start=start,
                                        end=next_end,
                                        text=region_text,
                                        ambiguity_type=AmbiguityType.OVERLAP,
                                        segmentations=segmentations,
                                        edges=edges_list,
                                        confidence=self._calculate_confidence(segmentations)
                                    )
                                    regions.append(region)
                                    for i in range(start, next_end):
                                        processed.add(i)
        
        return regions
    
    def _generate_cross_segmentations(
        self, 
        start: int, 
        end: int, 
        matches: Dict[int, List[Tuple[int, str]]], 
        text: str
    ) -> List[List[str]]:
        segmentations = []
        self._dfs_segmentations(start, end, matches, [], segmentations, text)
        return segmentations
    
    def _dfs_segmentations(
        self,
        current_pos: int,
        target_end: int,
        matches: Dict[int, List[Tuple[int, str]]],
        current_seg: List[str],
        all_segs: List[List[str]],
        text: str
    ) -> None:
        if current_pos == target_end:
            all_segs.append(current_seg.copy())
            return
        
        if current_pos > target_end:
            return
        
        if current_pos not in matches:
            self._dfs_segmentations(
                current_pos + 1, target_end, matches,
                current_seg + [text[current_pos]], all_segs, text
            )
            return
        
        edges = matches[current_pos]
        for next_end, word in edges:
            if next_end <= target_end:
                self._dfs_segmentations(
                    next_end, target_end, matches,
                    current_seg + [word], all_segs, text
                )
    
    def _build_segmentation(
        self,
        start: int,
        end: int,
        text: str,
        matches: Dict[int, List[Tuple[int, str]]]
    ) -> List[str]:
        segmentation = []
        pos = start
        
        while pos < end:
            if pos in matches:
                edges = matches[pos]
                valid_edges = [(e, w) for e, w in edges if e <= end]
                if valid_edges:
                    best_end, best_word = max(valid_edges, key=lambda x: x[0])
                    segmentation.append(best_word)
                    pos = best_end
                    continue
            
            segmentation.append(text[pos])
            pos += 1
        
        return segmentation
    
    def _calculate_confidence(self, segmentations: List[List[str]]) -> float:
        if len(segmentations) <= 1:
            return 1.0
        
        total_words = sum(len(seg) for seg in segmentations)
        avg_words = total_words / len(segmentations)
        
        variance = sum((len(seg) - avg_words) ** 2 for seg in segmentations) / len(segmentations)
        
        confidence = 1.0 / (1.0 + variance)
        
        return round(confidence, 4)
    
    def _merge_overlapping_regions(self, regions: List[AmbiguityRegion]) -> List[AmbiguityRegion]:
        if not regions:
            return []
        
        regions.sort(key=lambda r: (r.start, -r.end))
        
        merged = []
        for region in regions:
            if not merged:
                merged.append(region)
                continue
            
            last = merged[-1]
            
            if region.start >= last.end:
                merged.append(region)
            elif region.ambiguity_type == last.ambiguity_type:
                if region.end > last.end:
                    new_region = AmbiguityRegion(
                        start=last.start,
                        end=region.end,
                        text=last.text + region.text[last.end - region.start:],
                        ambiguity_type=last.ambiguity_type,
                        segmentations=last.segmentations + region.segmentations,
                        edges=list(set(last.edges + region.edges)),
                        confidence=min(last.confidence, region.confidence)
                    )
                    merged[-1] = new_region
        
        return merged
    
    def detect_from_lattice(self, lattice: 'Lattice') -> AmbiguityResult:
        regions = []
        
        for pos in range(lattice.length):
            outgoing = lattice.get_outgoing_edges(pos)
            
            if len(outgoing) <= 1:
                continue
            
            cross_edges = []
            combination_edges = []
            
            for i, edge1 in enumerate(outgoing):
                for edge2 in outgoing[i + 1:]:
                    if edge1.end != edge2.end:
                        cross_edges.append(edge1)
                        cross_edges.append(edge2)
            
            if cross_edges:
                cross_edges = list(set(cross_edges))
                max_end = max(e.end for e in cross_edges)
                region_text = lattice.text[pos:max_end]
                
                segmentations = self._get_lattice_segmentations(lattice, pos, max_end)
                
                region = AmbiguityRegion(
                    start=pos,
                    end=max_end,
                    text=region_text,
                    ambiguity_type=AmbiguityType.CROSS,
                    segmentations=segmentations,
                    edges=[(e.start, e.end, e.word) for e in cross_edges],
                    confidence=self._calculate_confidence(segmentations)
                )
                regions.append(region)
            else:
                combination_edges = outgoing
                max_end = max(e.end for e in combination_edges)
                region_text = lattice.text[pos:max_end]
                
                segmentations = [[e.word] for e in combination_edges]
                
                region = AmbiguityRegion(
                    start=pos,
                    end=max_end,
                    text=region_text,
                    ambiguity_type=AmbiguityType.COMBINATION,
                    segmentations=segmentations,
                    edges=[(e.start, e.end, e.word) for e in combination_edges],
                    confidence=self._calculate_confidence(segmentations)
                )
                regions.append(region)
        
        regions = self._merge_overlapping_regions(regions)
        
        cross_count = sum(1 for r in regions if r.ambiguity_type == AmbiguityType.CROSS)
        combination_count = sum(1 for r in regions if r.ambiguity_type == AmbiguityType.COMBINATION)
        overlap_count = sum(1 for r in regions if r.ambiguity_type == AmbiguityType.OVERLAP)
        
        return AmbiguityResult(
            text=lattice.text,
            total_ambiguities=len(regions),
            cross_count=cross_count,
            combination_count=combination_count,
            overlap_count=overlap_count,
            regions=regions
        )
    
    def _get_lattice_segmentations(self, lattice: 'Lattice', start: int, end: int) -> List[List[str]]:
        segmentations = []
        
        def dfs(current_pos: int, current_seg: List[str]):
            if current_pos == end:
                segmentations.append(current_seg.copy())
                return
            
            if current_pos > end:
                return
            
            outgoing = lattice.get_outgoing_edges(current_pos)
            
            if not outgoing:
                if current_pos < len(lattice.text):
                    dfs(current_pos + 1, current_seg + [lattice.text[current_pos]])
                return
            
            for edge in outgoing:
                if edge.end <= end:
                    dfs(edge.end, current_seg + [edge.word])
        
        dfs(start, [])
        return segmentations
    
    def get_ambiguity_statistics(self, text: str) -> Dict:
        result = self.detect(text)
        
        stats = {
            'text_length': len(text),
            'total_ambiguities': result.total_ambiguities,
            'ambiguity_density': result.total_ambiguities / len(text) if text else 0,
            'by_type': {
                'cross': result.cross_count,
                'combination': result.combination_count,
                'overlap': result.overlap_count
            },
            'avg_confidence': 0.0,
            'min_confidence': 1.0,
            'max_confidence': 0.0
        }
        
        if result.regions:
            confidences = [r.confidence for r in result.regions]
            stats['avg_confidence'] = sum(confidences) / len(confidences)
            stats['min_confidence'] = min(confidences)
            stats['max_confidence'] = max(confidences)
        
        return stats


__all__ = [
    'AmbiguityType',
    'AmbiguityRegion',
    'AmbiguityResult',
    'AmbiguityDetector'
]
