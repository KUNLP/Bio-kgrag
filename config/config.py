"""
Configuration file for KG-RAG Benchmark
All paths and settings should be configured here
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
SYNLETHDB_DIR = DATA_DIR / "synlethdb"
OUTPUT_DIR = BASE_DIR / "outputs"

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# LLM configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# Question generation targets
QUESTION_TARGETS = {
    "One-hop": 500,
    "Two-hop": 200,
    "Intersection": 100,
    "Attribute": 200
}

# PubMed configuration
PUBMED_MAX_DOCS = int(os.getenv("PUBMED_MAX_DOCS", "5"))

# Output file paths
QA_OUTPUT_FILE = OUTPUT_DIR / "qa_pairs.json"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediates"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
INTERMEDIATE_DIR.mkdir(exist_ok=True)

