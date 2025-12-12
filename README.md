# KG-RAG Benchmark

A Biomedical Benchmark for Large Language Models with Knowledge Graphs and Retrieval-Augmented Generation

This repository contains the implementation of the KG-RAG framework for automatically generating biomedical question-answer pairs using Knowledge Graphs (KG) and Retrieval-Augmented Generation (RAG).

## 📋 Overview

The KG-RAG framework combines:
- **Knowledge Graph (KG)**: Extracts structured entity relationships from SynLethDB
- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant PubMed literature
- **Large Language Models (LLM)**: Generates high-quality question-answer pairs

### Dataset Statistics
- **Total Questions**: 1,000 generated → 775 after quality filtering
- **Question Types**:
  - One-hop: 500 (50%)
  - Two-hop: 200 (20%)
  - Intersection: 100 (10%)
  - Attribute: 200 (20%)

## 🚀 Quick Start

### Prerequisites

1. **Neo4j Database**: Install and run Neo4j (version 5.0+)
2. **Python**: Python 3.8+
3. **SynLethDB Data**: Download SynLethDB CSV file

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kg-rag-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your configuration
```

### Configuration

Edit `.env` file with your settings:
- `NEO4J_URI`: Neo4j connection URI
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `OPENAI_API_KEY`: Your OpenAI API key
- `LLM_MODEL`: LLM model to use (default: gpt-3.5-turbo)

## 📖 Usage

### 1. Load SynLethDB Data to Neo4j

First, extract the schema and optionally load data:

```bash
python src/kg_loader.py \
    --csv path/to/synlethdb.csv \
    --schema data/synlethdb/schema.json \
    --force_load
```

### 2. Generate Question-Answer Pairs

Generate QA pairs using the KG-RAG framework:

```bash
python src/qa_generator.py
```

The generated QA pairs will be saved to `outputs/qa_pairs.json`.

### 3. Evaluate QA Pairs

Evaluate the generated QA pairs using GPT:

```bash
python evaluation/evaluators/gpt_evaluator.py \
    --input outputs/qa_pairs.json \
    --output outputs/evaluation_results/scores_gpt4.json \
    --model gpt-4
```

### 4. Calculate Agreement Ratio

Compare evaluation results from multiple models:

```bash
python evaluation/analysis/agreement_ratio.py \
    --files file1.json file2.json file3.json \
    --names model1 model2 model3 \
    --score-types naturalness_score answer_appropriateness_score
```

## 📁 Project Structure

```
kg-rag-benchmark/
├── config/
│   └── config.py              # Configuration settings
├── data/
│   └── synlethdb/
│       └── schema.json        # SynLethDB schema
├── src/
│   ├── kg_loader.py          # SynLethDB data loader
│   └── qa_generator.py       # QA pair generator (KG-RAG)
├── evaluation/
│   ├── evaluators/
│   │   └── gpt_evaluator.py  # GPT-based evaluator
│   └── analysis/
│       └── agreement_ratio.py # Agreement ratio calculator
├── outputs/                   # Generated outputs (gitignored)
├── requirements.txt
├── env.example
└── README.md
```

## 🔧 Question Types

### One-hop Questions
Direct relationship queries between two entities.
- Example: "Which types of cancer are associated with MET?"
- Neo4j Query: `MATCH (h)-[r]->(t) RETURN ...`

### Two-hop Questions
Multi-step relationship queries through intermediate entities.
- Example: "Which gene is expressed in the brain and regulates dopamine levels?"
- Neo4j Query: `MATCH (h)-[r1]->(m)-[r2]->(t) RETURN ...`

### Intersection Questions
Find common entities connected to multiple head entities.
- Example: "What is the common target of both EGFR and HER2 inhibitors?"
- Neo4j Query: `MATCH (h1)-[r1]->(c)<-[r2]-(h2) RETURN ...`

### Attribute Questions
Query entity attributes or descriptions.
- Example: "What is the molecular function of TP53?"
- Neo4j Query: `MATCH (e) WHERE e.description IS NOT NULL RETURN ...`

## 📊 Evaluation Metrics

The framework evaluates QA pairs on:
1. **Naturalness**: How natural the question reads (1-5 scale)
2. **Answer Appropriateness**: How well the answer matches the question (1-5 scale)
3. **Evidence Support**: PubMed evidence quality (1-5 scale)

## 🔬 Research

This implementation is based on the paper:
> "A Biomedical Benchmark for Large Language Models with Knowledge Graphs and Retrieval-Augmented Generation"

### Key Results
- **Average Quality Score**: 95.98%
- **Question Naturalness**: 99.48%
- **Answer Appropriateness**: 92.21%
- **Comparison with PcQA**: 4.2%p higher performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Specify your license here]

## 🙏 Acknowledgments

- SynLethDB for the knowledge graph data
- PubMed for literature retrieval
- OpenAI for language models

## 📧 Contact

[Your contact information]

