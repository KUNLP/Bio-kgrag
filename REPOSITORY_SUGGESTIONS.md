# GitHub 레포지토리 이름 및 설명 추천

## 🎯 추천 레포지토리 이름

### 1순위: `kg-rag-benchmark` (현재 이름)
- ✅ 간결하고 명확함
- ✅ KG-RAG 프레임워크를 직접 표현
- ✅ 벤치마크 목적이 명확함

### 2순위: `biomedical-kg-rag`
- ✅ 도메인(바이오메디컬) 명시
- ✅ KG-RAG 기술 스택 표현

### 3순위: `synlethdb-qa-benchmark`
- ✅ 사용 데이터셋(SynLethDB) 명시
- ✅ QA 벤치마크 목적 명확

## 📝 레포지토리 설명 (Description)

### 짧은 버전 (한 줄)
```
A biomedical benchmark dataset generator using Knowledge Graphs and Retrieval-Augmented Generation (KG-RAG)
```

### 중간 버전 (2-3줄)
```
KG-RAG: A framework for automatically generating biomedical question-answer pairs using Knowledge Graphs (SynLethDB) and Retrieval-Augmented Generation (PubMed). Generates 775 high-quality QA pairs for LLM evaluation.
```

### 상세 버전 (README용)
```
KG-RAG Benchmark: A Biomedical Benchmark for Large Language Models

Automatically generates high-quality biomedical question-answer pairs by combining:
- Knowledge Graphs (SynLethDB: 54K nodes, 2.2M edges)
- Retrieval-Augmented Generation (PubMed literature)
- Large Language Models (GPT-3.5/GPT-4)

Features:
- 4 question types: One-hop, Two-hop, Intersection, Attribute
- 775 validated QA pairs (from 1,000 generated)
- 95.98% average quality score
- Automated generation pipeline
```

## 🏷️ 추천 Topics/Tags

```
biomedical-nlp
knowledge-graph
retrieval-augmented-generation
question-answering
benchmark-dataset
synlethdb
pubmed
llm-evaluation
biomedical-ai
rag
neo4j
```

## 📋 README 상단 예시

```markdown
# KG-RAG Benchmark

> A Biomedical Benchmark for Large Language Models with Knowledge Graphs and Retrieval-Augmented Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Automatically generate high-quality biomedical question-answer pairs using Knowledge Graphs and RAG.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python src/qa_generator.py
```

## 📊 Dataset

- **Total**: 775 validated QA pairs
- **Question Types**: One-hop (50%), Two-hop (20%), Intersection (10%), Attribute (20%)
- **Quality Score**: 95.98% average
```

## 🎨 레포지토리 설정 예시

### Description (짧은 버전)
```
Biomedical QA benchmark generator using KG-RAG framework (SynLethDB + PubMed + LLM)
```

### Website (논문 링크가 있다면)
```
https://arxiv.org/abs/...
```

### Topics
```
biomedical-nlp, knowledge-graph, retrieval-augmented-generation, question-answering, benchmark-dataset, synlethdb, pubmed, llm-evaluation, biomedical-ai, rag, neo4j
```

## 💡 추가 제안

### 레포지토리 이름 옵션들
1. `kg-rag-benchmark` ⭐ (추천)
2. `biomedical-kg-rag`
3. `kg-rag-biomedical-benchmark`
4. `synlethdb-qa-generator`
5. `biomedical-qa-kg-rag`

### 설명 옵션들
1. **간결**: `Biomedical QA benchmark generator using KG-RAG (Knowledge Graph + RAG)`
2. **표준**: `A framework for automatically generating biomedical question-answer pairs using Knowledge Graphs and Retrieval-Augmented Generation`
3. **상세**: `KG-RAG Benchmark: Generate 775 high-quality biomedical QA pairs from SynLethDB knowledge graph and PubMed literature using LLMs`

