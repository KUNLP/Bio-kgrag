"""
SynLethDB Knowledge Graph Loader
Loads SynLethDB CSV data into Neo4j
"""
import pandas as pd
import json
from neo4j import GraphDatabase
from tqdm import tqdm
import logging
from typing import Dict
import argparse
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_schema(csv_file: str, schema_file: str) -> Dict:
    """Extract schema from SynLethDB CSV file"""
    logger.info(f"Extracting schema from {csv_file}")
    print(f"[INFO] Loading CSV file: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"[INFO] CSV loaded, rows: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        raise

    entity_types = df[df['_labels'].notna()]['_labels'].str.replace(':', '').unique().tolist()
    relation_types = df[df['_type'].notna()]['_type'].unique().tolist()
    print(f"[INFO] Entity types: {len(entity_types)}")
    print(f"[INFO] Relation types: {len(relation_types)}")

    schema = {
        "entity_types": entity_types,
        "relation_types": relation_types
    }
    
    try:
        schema_path = Path(schema_file)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=4)
        logger.info(f"Schema saved to {schema_file}")
    except Exception as e:
        logger.error(f"Failed to save schema: {str(e)}")
        raise

    return schema


def load_synlethdb(uri: str, user: str, password: str, csv_file: str, 
                   force_load: bool = False, batch_size: int = 1000) -> None:
    """Load SynLethDB data into Neo4j"""
    if not force_load:
        logger.info("Skipping data load. Use --force_load to load data.")
        return

    logger.info(f"Loading SynLethDB from {csv_file} to Neo4j")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        print("[INFO] Neo4j connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        raise

    try:
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"[INFO] CSV loaded, rows: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        raise

    with driver.session() as session:
        # Load edges
        edges = df[df['_start'].notna() & df['_end'].notna()]
        print(f"[INFO] Total edges to load: {len(edges)}")
        
        for i in tqdm(range(0, len(edges), batch_size), desc="Loading edges"):
            batch = edges.iloc[i:i+batch_size]
            tx = None
            try:
                tx = session.begin_transaction()
                for _, row in batch.iterrows():
                    tx.run("""
                        MATCH (s:Node {id: $source}), (t:Node {id: $target})
                        CALL apoc.create.relationship(s, $type, {}, t) YIELD rel
                        RETURN rel
                    """,
                    source=row['_start'], target=row['_end'], type=row['_type'])
                tx.commit()
            except Exception as e:
                logger.error(f"Failed to load edge batch: {str(e)}")
                if tx is not None:
                    tx.rollback()
                raise

    driver.close()
    logger.info("SynLethDB loading completed")


def parse_args():
    parser = argparse.ArgumentParser(description="SynLethDB Data Loader and Schema Extractor")
    parser.add_argument("--csv", type=str, help="Path to synlethdb.csv file")
    parser.add_argument("--schema", type=str, help="Path to output schema.json file")
    parser.add_argument("--uri", type=str, default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
                       help="Neo4j URI")
    parser.add_argument("--user", type=str, default=os.getenv("NEO4J_USER", "neo4j"), 
                       help="Neo4j username")
    parser.add_argument("--password", type=str, default=os.getenv("NEO4J_PASSWORD", "password"), 
                       help="Neo4j password")
    parser.add_argument("--force_load", action="store_true", 
                       help="Force load data to Neo4j")
    parser.add_argument("--batch_size", type=int, default=1000, 
                       help="Batch size for loading")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.csv and args.schema:
        extract_schema(args.csv, args.schema)
        if args.force_load:
            load_synlethdb(args.uri, args.user, args.password, args.csv, 
                          args.force_load, args.batch_size)

