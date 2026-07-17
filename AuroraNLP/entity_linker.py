"""
Entity Linking Module for AuroraNLP

This module provides entity linking functionality that connects named entities
recognized by NER to knowledge base entries, including entity normalization
and knowledge base integration.
"""

from typing import List, Dict, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os
import re
import pickle
from abc import ABC, abstractmethod

from .ner import Entity, NERRecognizer, NER_ENTITY_TYPES


@dataclass
class KnowledgeEntity:
    """
    Represents an entity in the knowledge base.
    
    Attributes:
        entity_id: Unique identifier for the entity
        canonical_name: The canonical/official name of the entity
        entity_type: Type of entity (PER, LOC, ORG, etc.)
        aliases: Alternative names/aliases for the entity
        attributes: Additional attributes and metadata
        description: Brief description of the entity
        confidence: Confidence score for the entity entry
        source: Source of the knowledge (e.g., 'wikidata', 'custom')
        external_ids: External reference IDs (e.g., Wikidata Q-ID)
    """
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    confidence: float = 1.0
    source: str = "custom"
    external_ids: Dict[str, str] = field(default_factory=dict)
    
    def add_alias(self, alias: str):
        if alias not in self.aliases and alias != self.canonical_name:
            self.aliases.append(alias)
    
    def remove_alias(self, alias: str):
        if alias in self.aliases:
            self.aliases.remove(alias)
    
    def get_all_names(self) -> List[str]:
        names = [self.canonical_name]
        names.extend(self.aliases)
        return names
    
    def matches(self, text: str, case_sensitive: bool = False) -> bool:
        if not case_sensitive:
            text = text.lower()
            if text == self.canonical_name.lower():
                return True
            return any(text == alias.lower() for alias in self.aliases)
        return text == self.canonical_name or text in self.aliases
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'canonical_name': self.canonical_name,
            'entity_type': self.entity_type,
            'aliases': self.aliases,
            'attributes': self.attributes,
            'description': self.description,
            'confidence': self.confidence,
            'source': self.source,
            'external_ids': self.external_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntity':
        return cls(
            entity_id=data['entity_id'],
            canonical_name=data['canonical_name'],
            entity_type=data['entity_type'],
            aliases=data.get('aliases', []),
            attributes=data.get('attributes', {}),
            description=data.get('description', ''),
            confidence=data.get('confidence', 1.0),
            source=data.get('source', 'custom'),
            external_ids=data.get('external_ids', {})
        )
    
    def __repr__(self) -> str:
        return f"KnowledgeEntity({self.entity_id}, '{self.canonical_name}', {self.entity_type})"


@dataclass
class LinkedEntity:
    """
    Represents an entity that has been linked to a knowledge base entry.
    
    Attributes:
        entity: The original NER entity
        knowledge_entity: The linked knowledge base entity (None if not linked)
        confidence: Confidence score for the linking
        candidate_entities: List of candidate entities with their scores
        is_linked: Whether the entity was successfully linked
        linking_method: Method used for linking ('exact', 'fuzzy', 'context')
    """
    entity: Entity
    knowledge_entity: Optional[KnowledgeEntity] = None
    confidence: float = 0.0
    candidate_entities: List[Tuple[KnowledgeEntity, float]] = field(default_factory=list)
    is_linked: bool = False
    linking_method: str = ""
    
    def get_canonical_name(self) -> str:
        if self.knowledge_entity:
            return self.knowledge_entity.canonical_name
        return self.entity.text
    
    def get_entity_id(self) -> Optional[str]:
        if self.knowledge_entity:
            return self.knowledge_entity.entity_id
        return None
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        if self.knowledge_entity and key in self.knowledge_entity.attributes:
            return self.knowledge_entity.attributes[key]
        return default
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'text': self.entity.text,
            'type': self.entity.entity_type,
            'start': self.entity.start,
            'end': self.entity.end,
            'is_linked': self.is_linked,
            'confidence': self.confidence,
            'linking_method': self.linking_method,
            'canonical_name': self.get_canonical_name(),
            'entity_id': self.get_entity_id()
        }
        
        if self.knowledge_entity:
            result['knowledge_entity'] = self.knowledge_entity.to_dict()
        
        if self.candidate_entities:
            result['candidates'] = [
                {'entity': ke.to_dict(), 'score': score}
                for ke, score in self.candidate_entities[:5]
            ]
        
        return result
    
    def __repr__(self) -> str:
        status = "linked" if self.is_linked else "unlinked"
        return f"LinkedEntity('{self.entity.text}', {status}, conf={self.confidence:.2f})"


class KnowledgeBase:
    """
    Knowledge base for storing and querying entity information.
    
    Supports:
    - Entity storage and retrieval by ID or name
    - Alias-based lookup
    - Type-based filtering
    - Persistence (save/load)
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._entities: Dict[str, KnowledgeEntity] = {}
        self._name_index: Dict[str, Set[str]] = defaultdict(set)
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._alias_index: Dict[str, Set[str]] = defaultdict(set)
        self._next_id = 1
    
    def _generate_id(self) -> str:
        entity_id = f"KB_{self._next_id:06d}"
        self._next_id += 1
        return entity_id
    
    def add_entity(
        self,
        canonical_name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        description: str = "",
        confidence: float = 1.0,
        source: str = "custom",
        external_ids: Optional[Dict[str, str]] = None,
        entity_id: Optional[str] = None
    ) -> KnowledgeEntity:
        if entity_id is None:
            entity_id = self._generate_id()
        
        entity = KnowledgeEntity(
            entity_id=entity_id,
            canonical_name=canonical_name,
            entity_type=entity_type,
            aliases=aliases or [],
            attributes=attributes or {},
            description=description,
            confidence=confidence,
            source=source,
            external_ids=external_ids or {}
        )
        
        return self._add_entity_object(entity)
    
    def _add_entity_object(self, entity: KnowledgeEntity) -> KnowledgeEntity:
        if entity.entity_id in self._entities:
            self._remove_from_indexes(self._entities[entity.entity_id])
        
        self._entities[entity.entity_id] = entity
        
        name_lower = entity.canonical_name.lower()
        self._name_index[name_lower].add(entity.entity_id)
        self._type_index[entity.entity_type].add(entity.entity_id)
        
        for alias in entity.aliases:
            alias_lower = alias.lower()
            self._alias_index[alias_lower].add(entity.entity_id)
        
        return entity
    
    def _remove_from_indexes(self, entity: KnowledgeEntity):
        name_lower = entity.canonical_name.lower()
        self._name_index[name_lower].discard(entity.entity_id)
        self._type_index[entity.entity_type].discard(entity.entity_id)
        
        for alias in entity.aliases:
            alias_lower = alias.lower()
            self._alias_index[alias_lower].discard(entity.entity_id)
    
    def get_entity_by_id(self, entity_id: str) -> Optional[KnowledgeEntity]:
        return self._entities.get(entity_id)
    
    def get_entities_by_name(
        self,
        name: str,
        case_sensitive: bool = False
    ) -> List[KnowledgeEntity]:
        if case_sensitive:
            entities = []
            for entity in self._entities.values():
                if entity.canonical_name == name or name in entity.aliases:
                    entities.append(entity)
            return entities
        
        name_lower = name.lower()
        entity_ids = set()
        
        entity_ids.update(self._name_index.get(name_lower, set()))
        entity_ids.update(self._alias_index.get(name_lower, set()))
        
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def get_entities_by_type(self, entity_type: str) -> List[KnowledgeEntity]:
        entity_ids = self._type_index.get(entity_type, set())
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[KnowledgeEntity, float]]:
        query_lower = query.lower()
        results: List[Tuple[KnowledgeEntity, float]] = []
        
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            
            score = 0.0
            
            if entity.canonical_name.lower() == query_lower:
                score = 1.0
            elif query_lower in entity.canonical_name.lower():
                score = 0.8
            else:
                for alias in entity.aliases:
                    alias_lower = alias.lower()
                    if alias_lower == query_lower:
                        score = max(score, 0.9)
                    elif query_lower in alias_lower:
                        score = max(score, 0.7)
            
            if score > 0:
                results.append((entity, score))
        
        results.sort(key=lambda x: (-x[1], x[0].canonical_name))
        return results[:limit]
    
    def fuzzy_search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        threshold: float = 0.6,
        limit: int = 10
    ) -> List[Tuple[KnowledgeEntity, float]]:
        from difflib import SequenceMatcher
        
        query_lower = query.lower()
        results: List[Tuple[KnowledgeEntity, float]] = []
        
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            
            best_score = 0.0
            
            names_to_check = [entity.canonical_name] + entity.aliases
            for name in names_to_check:
                name_lower = name.lower()
                ratio = SequenceMatcher(None, query_lower, name_lower).ratio()
                best_score = max(best_score, ratio)
            
            if best_score >= threshold:
                results.append((entity, best_score))
        
        results.sort(key=lambda x: (-x[1], x[0].canonical_name))
        return results[:limit]
    
    def remove_entity(self, entity_id: str) -> bool:
        if entity_id not in self._entities:
            return False
        
        entity = self._entities[entity_id]
        self._remove_from_indexes(entity)
        del self._entities[entity_id]
        return True
    
    def update_entity(
        self,
        entity_id: str,
        canonical_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> Optional[KnowledgeEntity]:
        if entity_id not in self._entities:
            return None
        
        entity = self._entities[entity_id]
        
        self._remove_from_indexes(entity)
        
        if canonical_name is not None:
            entity.canonical_name = canonical_name
        if aliases is not None:
            entity.aliases = aliases
        if attributes is not None:
            entity.attributes.update(attributes)
        if description is not None:
            entity.description = description
        
        self._name_index[entity.canonical_name.lower()].add(entity.entity_id)
        self._type_index[entity.entity_type].add(entity.entity_id)
        for alias in entity.aliases:
            self._alias_index[alias.lower()].add(entity.entity_id)
        
        return entity
    
    def get_all_entities(self) -> List[KnowledgeEntity]:
        return list(self._entities.values())
    
    def get_entity_count(self) -> int:
        return len(self._entities)
    
    def get_entity_count_by_type(self) -> Dict[str, int]:
        return {etype: len(ids) for etype, ids in self._type_index.items()}
    
    def clear(self):
        self._entities.clear()
        self._name_index.clear()
        self._type_index.clear()
        self._alias_index.clear()
        self._next_id = 1
    
    def save(self, filepath: str):
        data = {
            'name': self.name,
            'next_id': self._next_id,
            'entities': [e.to_dict() for e in self._entities.values()]
        }
        
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.clear()
        self.name = data.get('name', self.name)
        self._next_id = data.get('next_id', 1)
        
        for entity_data in data.get('entities', []):
            entity = KnowledgeEntity.from_dict(entity_data)
            self._add_entity_object(entity)
    
    def merge(self, other: 'KnowledgeBase', overwrite: bool = False):
        for entity in other.get_all_entities():
            existing = self.get_entity_by_id(entity.entity_id)
            if existing and not overwrite:
                new_entity = KnowledgeEntity(
                    entity_id=self._generate_id(),
                    canonical_name=entity.canonical_name,
                    entity_type=entity.entity_type,
                    aliases=entity.aliases.copy(),
                    attributes=entity.attributes.copy(),
                    description=entity.description,
                    confidence=entity.confidence,
                    source=entity.source,
                    external_ids=entity.external_ids.copy()
                )
                self._add_entity_object(new_entity)
            else:
                if existing:
                    self._remove_from_indexes(existing)
                self._add_entity_object(entity)
    
    def export_to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'entities': [e.to_dict() for e in self._entities.values()],
            'statistics': {
                'total_entities': self.get_entity_count(),
                'by_type': self.get_entity_count_by_type()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeBase':
        kb = cls(name=data.get('name', 'default'))
        for entity_data in data.get('entities', []):
            entity = KnowledgeEntity.from_dict(entity_data)
            kb._add_entity_object(entity)
        return kb
    
    def __len__(self) -> int:
        return len(self._entities)
    
    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._entities
    
    def __repr__(self) -> str:
        return f"KnowledgeBase('{self.name}', entities={len(self._entities)})"


class EntityNormalizer:
    """
    Entity normalizer for standardizing entity names.
    
    Features:
    - Alias mapping
    - Abbreviation expansion
    - Name standardization
    - Custom normalization rules
    """
    
    def __init__(self):
        self._alias_map: Dict[str, str] = {}
        self._abbreviations: Dict[str, str] = {}
        self._normalization_rules: List[Tuple[str, str]] = []
        self._type_specific_rules: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    def add_alias(self, alias: str, canonical: str):
        self._alias_map[alias.lower()] = canonical
    
    def add_aliases(self, aliases: Dict[str, str]):
        for alias, canonical in aliases.items():
            self.add_alias(alias, canonical)
    
    def add_abbreviation(self, abbr: str, full_form: str):
        self._abbreviations[abbr.lower()] = full_form
    
    def add_abbreviations(self, abbreviations: Dict[str, str]):
        for abbr, full_form in abbreviations.items():
            self.add_abbreviation(abbr, full_form)
    
    def add_normalization_rule(self, pattern: str, replacement: str):
        compiled = re.compile(pattern)
        self._normalization_rules.append((compiled, replacement))
    
    def add_type_specific_rule(self, entity_type: str, pattern: str, replacement: str):
        compiled = re.compile(pattern)
        self._type_specific_rules[entity_type].append((compiled, replacement))
    
    def normalize(self, text: str, entity_type: Optional[str] = None) -> str:
        normalized = text.strip()
        
        text_lower = normalized.lower()
        if text_lower in self._alias_map:
            return self._alias_map[text_lower]
        
        if text_lower in self._abbreviations:
            normalized = self._abbreviations[text_lower]
        
        for pattern, replacement in self._normalization_rules:
            normalized = pattern.sub(replacement, normalized)
        
        if entity_type and entity_type in self._type_specific_rules:
            for pattern, replacement in self._type_specific_rules[entity_type]:
                normalized = pattern.sub(replacement, normalized)
        
        return normalized
    
    def get_canonical(self, text: str) -> Optional[str]:
        return self._alias_map.get(text.lower())
    
    def expand_abbreviation(self, text: str) -> str:
        return self._abbreviations.get(text.lower(), text)
    
    def clear(self):
        self._alias_map.clear()
        self._abbreviations.clear()
        self._normalization_rules.clear()
        self._type_specific_rules.clear()
    
    def save(self, filepath: str):
        data = {
            'alias_map': self._alias_map,
            'abbreviations': self._abbreviations,
            'normalization_rules': [(p.pattern, r) for p, r in self._normalization_rules],
            'type_specific_rules': {
                etype: [(p.pattern, r) for p, r in rules]
                for etype, rules in self._type_specific_rules.items()
            }
        }
        
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.clear()
        self._alias_map = data.get('alias_map', {})
        self._abbreviations = data.get('abbreviations', {})
        
        for pattern, replacement in data.get('normalization_rules', []):
            self.add_normalization_rule(pattern, replacement)
        
        for etype, rules in data.get('type_specific_rules', {}).items():
            for pattern, replacement in rules:
                self.add_type_specific_rule(etype, pattern, replacement)


class EntityLinker:
    """
    Main entity linking interface that connects NER entities to knowledge base.
    
    Features:
    - Exact matching
    - Fuzzy matching
    - Context-based disambiguation
    - Candidate ranking
    - Batch processing
    """
    
    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        ner_recognizer: Optional[NERRecognizer] = None,
        normalizer: Optional[EntityNormalizer] = None
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.ner_recognizer = ner_recognizer
        self.normalizer = normalizer or EntityNormalizer()
        
        self._fuzzy_threshold = 0.7
        self._context_window = 50
        self._max_candidates = 5
    
    def set_fuzzy_threshold(self, threshold: float):
        self._fuzzy_threshold = threshold
    
    def set_context_window(self, window: int):
        self._context_window = window
    
    def set_max_candidates(self, max_candidates: int):
        self._max_candidates = max_candidates
    
    def link_entity(
        self,
        entity: Entity,
        context: Optional[str] = None
    ) -> LinkedEntity:
        normalized_name = self.normalizer.normalize(entity.text, entity.entity_type)
        
        candidates = self._find_candidates(entity, normalized_name)
        
        if not candidates:
            return LinkedEntity(
                entity=entity,
                confidence=0.0,
                is_linked=False
            )
        
        if context:
            candidates = self._disambiguate_by_context(entity, candidates, context)
        
        best_entity, best_score = candidates[0]
        
        return LinkedEntity(
            entity=entity,
            knowledge_entity=best_entity,
            confidence=best_score,
            candidate_entities=candidates[:self._max_candidates],
            is_linked=True,
            linking_method='exact' if best_score >= 0.95 else 'fuzzy'
        )
    
    def _find_candidates(
        self,
        entity: Entity,
        normalized_name: str
    ) -> List[Tuple[KnowledgeEntity, float]]:
        candidates: List[Tuple[KnowledgeEntity, float]] = []
        
        exact_matches = self.knowledge_base.get_entities_by_name(normalized_name)
        for ke in exact_matches:
            if ke.entity_type == entity.entity_type:
                candidates.append((ke, 1.0))
            elif ke.entity_type != entity.entity_type:
                candidates.append((ke, 0.8))
        
        if not candidates:
            fuzzy_matches = self.knowledge_base.fuzzy_search(
                normalized_name,
                entity_type=entity.entity_type,
                threshold=self._fuzzy_threshold,
                limit=self._max_candidates
            )
            candidates.extend(fuzzy_matches)
        
        if not candidates:
            fuzzy_matches = self.knowledge_base.fuzzy_search(
                normalized_name,
                threshold=self._fuzzy_threshold,
                limit=self._max_candidates
            )
            for ke, score in fuzzy_matches:
                adjusted_score = score * 0.8 if ke.entity_type != entity.entity_type else score
                candidates.append((ke, adjusted_score))
        
        candidates.sort(key=lambda x: -x[1])
        return candidates
    
    def _disambiguate_by_context(
        self,
        entity: Entity,
        candidates: List[Tuple[KnowledgeEntity, float]],
        context: str
    ) -> List[Tuple[KnowledgeEntity, float]]:
        context_lower = context.lower()
        
        rescored = []
        for ke, base_score in candidates:
            context_score = base_score
            
            if ke.description:
                desc_lower = ke.description.lower()
                if entity.text.lower() in desc_lower:
                    context_score += 0.1
            
            for attr_key, attr_value in ke.attributes.items():
                if isinstance(attr_value, str):
                    if attr_value.lower() in context_lower:
                        context_score += 0.05
            
            rescored.append((ke, min(context_score, 1.0)))
        
        rescored.sort(key=lambda x: -x[1])
        return rescored
    
    def link_entities(
        self,
        entities: List[Entity],
        context: Optional[str] = None
    ) -> List[LinkedEntity]:
        return [self.link_entity(entity, context) for entity in entities]
    
    def link_text(
        self,
        text: str,
        use_ner: bool = True
    ) -> List[LinkedEntity]:
        if use_ner and self.ner_recognizer:
            if not self.ner_recognizer.is_trained():
                raise RuntimeError("NER recognizer has not been trained. Call train() first.")
            entities = self.ner_recognizer.recognize(text)
        else:
            entities = []
        
        return self.link_entities(entities, text)
    
    def batch_link(
        self,
        texts: List[str],
        use_ner: bool = True
    ) -> List[List[LinkedEntity]]:
        return [self.link_text(text, use_ner) for text in texts]
    
    def add_knowledge_entry(
        self,
        canonical_name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        description: str = "",
        external_ids: Optional[Dict[str, str]] = None
    ) -> KnowledgeEntity:
        return self.knowledge_base.add_entity(
            canonical_name=canonical_name,
            entity_type=entity_type,
            aliases=aliases,
            attributes=attributes,
            description=description,
            external_ids=external_ids
        )
    
    def add_normalization_alias(self, alias: str, canonical: str):
        self.normalizer.add_alias(alias, canonical)
    
    def get_linked_entities_by_type(
        self,
        linked_entities: List[LinkedEntity],
        entity_type: str
    ) -> List[LinkedEntity]:
        return [
            le for le in linked_entities
            if le.is_linked and le.knowledge_entity
            and le.knowledge_entity.entity_type == entity_type
        ]
    
    def get_unlinked_entities(
        self,
        linked_entities: List[LinkedEntity]
    ) -> List[LinkedEntity]:
        return [le for le in linked_entities if not le.is_linked]
    
    def annotate_text(
        self,
        text: str,
        use_ner: bool = True
    ) -> str:
        linked_entities = self.link_text(text, use_ner)
        
        if not linked_entities:
            return text
        
        linked_entities = sorted(linked_entities, key=lambda le: le.entity.start)
        
        result = []
        last_end = 0
        
        for le in linked_entities:
            result.append(text[last_end:le.entity.start])
            
            if le.is_linked and le.knowledge_entity:
                annotation = f"[{le.entity.text}→{le.knowledge_entity.canonical_name}/{le.knowledge_entity.entity_type}]"
            else:
                annotation = f"[{le.entity.text}/{le.entity.entity_type}(未链接)]"
            
            result.append(annotation)
            last_end = le.entity.end
        
        result.append(text[last_end:])
        
        return ''.join(result)
    
    def get_statistics(
        self,
        linked_entities: List[LinkedEntity]
    ) -> Dict[str, Any]:
        total = len(linked_entities)
        linked = sum(1 for le in linked_entities if le.is_linked)
        
        by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {'linked': 0, 'unlinked': 0})
        for le in linked_entities:
            etype = le.entity.entity_type
            if le.is_linked:
                by_type[etype]['linked'] += 1
            else:
                by_type[etype]['unlinked'] += 1
        
        avg_confidence = 0.0
        if linked > 0:
            avg_confidence = sum(
                le.confidence for le in linked_entities if le.is_linked
            ) / linked
        
        return {
            'total_entities': total,
            'linked_entities': linked,
            'unlinked_entities': total - linked,
            'link_rate': linked / total if total > 0 else 0.0,
            'average_confidence': avg_confidence,
            'by_type': dict(by_type)
        }
    
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        
        kb_path = os.path.join(directory, 'knowledge_base.json')
        self.knowledge_base.save(kb_path)
        
        norm_path = os.path.join(directory, 'normalizer.json')
        self.normalizer.save(norm_path)
    
    def load(self, directory: str):
        kb_path = os.path.join(directory, 'knowledge_base.json')
        if os.path.exists(kb_path):
            self.knowledge_base.load(kb_path)
        
        norm_path = os.path.join(directory, 'normalizer.json')
        if os.path.exists(norm_path):
            self.normalizer.load(norm_path)


def create_sample_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase(name="sample")
    
    kb.add_entity(
        canonical_name="阿里巴巴集团",
        entity_type="ORG",
        aliases=["阿里巴巴", "阿里", "Alibaba", "Alibaba Group"],
        attributes={
            "industry": "电子商务",
            "founded": "1999",
            "headquarters": "杭州"
        },
        description="中国领先的电子商务公司",
        external_ids={"wikidata": "Q418249"}
    )
    
    kb.add_entity(
        canonical_name="腾讯控股有限公司",
        entity_type="ORG",
        aliases=["腾讯", "腾讯公司", "Tencent"],
        attributes={
            "industry": "互联网",
            "founded": "1998",
            "headquarters": "深圳"
        },
        description="中国互联网科技公司",
        external_ids={"wikidata": "Q860479"}
    )
    
    kb.add_entity(
        canonical_name="清华大学",
        entity_type="ORG",
        aliases=["清华", "THU"],
        attributes={
            "type": "大学",
            "location": "北京",
            "founded": "1911"
        },
        description="中国著名高等学府",
        external_ids={"wikidata": "Q16955"}
    )
    
    kb.add_entity(
        canonical_name="北京大学",
        entity_type="ORG",
        aliases=["北大", "PKU", "京师大学堂"],
        attributes={
            "type": "大学",
            "location": "北京",
            "founded": "1898"
        },
        description="中国著名高等学府",
        external_ids={"wikidata": "Q16952"}
    )
    
    kb.add_entity(
        canonical_name="北京市",
        entity_type="LOC",
        aliases=["北京", "北平", "Beijing", "Peking"],
        attributes={
            "type": "直辖市",
            "country": "中国"
        },
        external_ids={"wikidata": "Q956"}
    )
    
    kb.add_entity(
        canonical_name="上海市",
        entity_type="LOC",
        aliases=["上海", "Shanghai"],
        attributes={
            "type": "直辖市",
            "country": "中国"
        },
        external_ids={"wikidata": "Q8686"}
    )
    
    kb.add_entity(
        canonical_name="深圳市",
        entity_type="LOC",
        aliases=["深圳", "Shenzhen"],
        attributes={
            "type": "地级市",
            "province": "广东"
        },
        external_ids={"wikidata": "Q15175"}
    )
    
    return kb


def create_sample_normalizer() -> EntityNormalizer:
    normalizer = EntityNormalizer()
    
    normalizer.add_aliases({
        "阿里": "阿里巴巴集团",
        "阿里巴巴": "阿里巴巴集团",
        "腾讯": "腾讯控股有限公司",
        "腾讯公司": "腾讯控股有限公司",
        "清华": "清华大学",
        "北大": "北京大学",
    })
    
    normalizer.add_abbreviations({
        "THU": "清华大学",
        "PKU": "北京大学",
    })
    
    normalizer.add_normalization_rule(r'\s+', '')
    normalizer.add_normalization_rule(r'[（(].*?[）)]', '')
    
    normalizer.add_type_specific_rule('ORG', r'有限公司$', '集团')
    normalizer.add_type_specific_rule('LOC', r'^中国', '')
    
    return normalizer


__all__ = [
    'KnowledgeEntity',
    'LinkedEntity',
    'KnowledgeBase',
    'EntityNormalizer',
    'EntityLinker',
    'create_sample_knowledge_base',
    'create_sample_normalizer',
]
