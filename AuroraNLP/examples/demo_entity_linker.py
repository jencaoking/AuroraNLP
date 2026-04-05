"""
Entity Linking Demo for AuroraNLP

This demo showcases the entity linking functionality including:
- Knowledge base management
- Entity normalization
- Entity linking with NER integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP.entity_linker import (
    KnowledgeBase,
    KnowledgeEntity,
    EntityNormalizer,
    EntityLinker,
    LinkedEntity,
    create_sample_knowledge_base,
    create_sample_normalizer,
)
from AuroraNLP.ner import Entity, NERRecognizer, create_sample_ner_corpus


def demo_knowledge_base():
    print("=" * 60)
    print("1. Knowledge Base Demo")
    print("=" * 60)
    
    kb = create_sample_knowledge_base()
    
    print(f"\nKnowledge Base: {kb.name}")
    print(f"Total entities: {len(kb)}")
    
    print("\nEntity counts by type:")
    for etype, count in kb.get_entity_count_by_type().items():
        print(f"  {etype}: {count}")
    
    print("\n--- Search Examples ---")
    
    results = kb.search_entities("阿里")
    print(f"\nSearch '阿里':")
    for entity, score in results:
        print(f"  {entity.canonical_name} ({entity.entity_type}) - score: {score:.2f}")
    
    results = kb.fuzzy_search("阿里巴", threshold=0.6)
    print(f"\nFuzzy search '阿里巴':")
    for entity, score in results:
        print(f"  {entity.canonical_name} ({entity.entity_type}) - score: {score:.2f}")
    
    print("\n--- Get Entity by Name ---")
    entities = kb.get_entities_by_name("清华")
    for entity in entities:
        print(f"  Found: {entity.canonical_name}")
        print(f"    Aliases: {entity.aliases}")
        print(f"    Attributes: {entity.attributes}")
        print(f"    External IDs: {entity.external_ids}")


def demo_entity_normalizer():
    print("\n" + "=" * 60)
    print("2. Entity Normalizer Demo")
    print("=" * 60)
    
    normalizer = create_sample_normalizer()
    
    test_cases = [
        ("阿里", None),
        ("阿里巴巴", None),
        ("腾讯", None),
        ("清华", None),
        ("北大", None),
        ("THU", None),
        ("PKU", None),
        ("测试有限公司", "ORG"),
        ("中国北京", "LOC"),
    ]
    
    print("\nNormalization examples:")
    for text, entity_type in test_cases:
        normalized = normalizer.normalize(text, entity_type)
        type_str = f" ({entity_type})" if entity_type else ""
        print(f"  '{text}'{type_str} -> '{normalized}'")


def demo_entity_linking():
    print("\n" + "=" * 60)
    print("3. Entity Linking Demo")
    print("=" * 60)
    
    kb = create_sample_knowledge_base()
    normalizer = create_sample_normalizer()
    linker = EntityLinker(knowledge_base=kb, normalizer=normalizer)
    
    test_entities = [
        Entity("阿里巴巴", "ORG", 0, 4),
        Entity("阿里", "ORG", 0, 2),
        Entity("腾讯", "ORG", 0, 2),
        Entity("清华", "ORG", 0, 2),
        Entity("北京", "LOC", 0, 2),
        Entity("上海", "LOC", 0, 2),
        Entity("未知公司", "ORG", 0, 4),
    ]
    
    print("\nLinking entities:")
    for entity in test_entities:
        linked = linker.link_entity(entity)
        
        if linked.is_linked:
            print(f"\n  '{entity.text}' -> '{linked.get_canonical_name()}'")
            print(f"    Type: {entity.entity_type}")
            print(f"    Confidence: {linked.confidence:.2f}")
            print(f"    Method: {linked.linking_method}")
            if linked.knowledge_entity:
                print(f"    ID: {linked.knowledge_entity.entity_id}")
                if linked.knowledge_entity.attributes:
                    print(f"    Attributes: {linked.knowledge_entity.attributes}")
        else:
            print(f"\n  '{entity.text}' -> [UNLINKED]")
            print(f"    Type: {entity.entity_type}")


def demo_batch_linking():
    print("\n" + "=" * 60)
    print("4. Batch Linking Demo")
    print("=" * 60)
    
    kb = create_sample_knowledge_base()
    normalizer = create_sample_normalizer()
    linker = EntityLinker(knowledge_base=kb, normalizer=normalizer)
    
    entities = [
        Entity("阿里巴巴", "ORG", 0, 4),
        Entity("北京", "LOC", 5, 7),
        Entity("腾讯", "ORG", 8, 10),
        Entity("上海", "LOC", 11, 13),
        Entity("未知实体", "ORG", 14, 18),
    ]
    
    linked_entities = linker.link_entities(entities)
    
    print("\nBatch linking results:")
    stats = linker.get_statistics(linked_entities)
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Linked: {stats['linked_entities']}")
    print(f"  Unlinked: {stats['unlinked_entities']}")
    print(f"  Link rate: {stats['link_rate']:.1%}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")
    
    print("\n  By type:")
    for etype, counts in stats['by_type'].items():
        print(f"    {etype}: {counts['linked']} linked, {counts['unlinked']} unlinked")


def demo_annotation():
    print("\n" + "=" * 60)
    print("5. Text Annotation Demo")
    print("=" * 60)
    
    kb = create_sample_knowledge_base()
    normalizer = create_sample_normalizer()
    linker = EntityLinker(knowledge_base=kb, normalizer=normalizer)
    
    entities = [
        Entity("阿里巴巴", "ORG", 0, 4),
        Entity("北京", "LOC", 6, 8),
        Entity("腾讯", "ORG", 10, 12),
    ]
    
    text = "阿里巴巴在北京，腾讯在深圳"
    
    linked = linker.link_entities(entities, text)
    
    result = []
    last_end = 0
    for le in sorted(linked, key=lambda x: x.entity.start):
        result.append(text[last_end:le.entity.start])
        if le.is_linked:
            result.append(f"[{le.entity.text}→{le.knowledge_entity.canonical_name}/{le.knowledge_entity.entity_type}]")
        else:
            result.append(f"[{le.entity.text}/{le.entity.entity_type}(未链接)]")
        last_end = le.entity.end
    result.append(text[last_end:])
    
    annotated = ''.join(result)
    
    print(f"\nOriginal text: {text}")
    print(f"Annotated:     {annotated}")


def demo_custom_knowledge_base():
    print("\n" + "=" * 60)
    print("6. Custom Knowledge Base Demo")
    print("=" * 60)
    
    kb = KnowledgeBase(name="custom")
    
    kb.add_entity(
        canonical_name="百度在线网络技术(北京)有限公司",
        entity_type="ORG",
        aliases=["百度", "Baidu", "百度公司"],
        attributes={
            "industry": "互联网",
            "founded": "2000",
            "headquarters": "北京"
        },
        description="中国领先的搜索引擎公司",
        external_ids={"wikidata": "Q41753"}
    )
    
    kb.add_entity(
        canonical_name="杭州市",
        entity_type="LOC",
        aliases=["杭州", "Hangzhou"],
        attributes={
            "type": "省会城市",
            "province": "浙江"
        }
    )
    
    print(f"\nCreated custom knowledge base: {kb.name}")
    print(f"Total entities: {len(kb)}")
    
    normalizer = EntityNormalizer()
    normalizer.add_alias("百度", "百度在线网络技术(北京)有限公司")
    normalizer.add_alias("杭州", "杭州市")
    
    linker = EntityLinker(knowledge_base=kb, normalizer=normalizer)
    
    entity = Entity("百度", "ORG", 0, 2)
    linked = linker.link_entity(entity)
    
    print(f"\nLinking '百度':")
    if linked.is_linked:
        print(f"  Canonical: {linked.get_canonical_name()}")
        print(f"  Confidence: {linked.confidence:.2f}")
        print(f"  Attributes: {linked.knowledge_entity.attributes}")


def demo_persistence():
    print("\n" + "=" * 60)
    print("7. Persistence Demo")
    print("=" * 60)
    
    import tempfile
    import os
    
    kb = create_sample_knowledge_base()
    normalizer = create_sample_normalizer()
    linker = EntityLinker(knowledge_base=kb, normalizer=normalizer)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nSaving to: {tmpdir}")
        linker.save(tmpdir)
        
        print("Files created:")
        for filename in os.listdir(tmpdir):
            filepath = os.path.join(tmpdir, filename)
            size = os.path.getsize(filepath)
            print(f"  {filename}: {size} bytes")
        
        new_linker = EntityLinker()
        new_linker.load(tmpdir)
        
        print(f"\nLoaded knowledge base: {new_linker.knowledge_base.name}")
        print(f"Entities loaded: {len(new_linker.knowledge_base)}")
        
        entity = Entity("阿里巴巴", "ORG", 0, 4)
        linked = new_linker.link_entity(entity)
        
        print(f"\nTest linking after load:")
        print(f"  '阿里巴巴' -> '{linked.get_canonical_name()}'")
        print(f"  Confidence: {linked.confidence:.2f}")


def main():
    print("\n" + "=" * 60)
    print("AuroraNLP Entity Linking Demo")
    print("=" * 60)
    
    demo_knowledge_base()
    demo_entity_normalizer()
    demo_entity_linking()
    demo_batch_linking()
    demo_annotation()
    demo_custom_knowledge_base()
    demo_persistence()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
