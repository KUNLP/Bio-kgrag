"""
KG-RAG Question-Answer Pair Generator
Generates biomedical QA pairs using Knowledge Graph and RAG
"""
import os
import json
import random
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PubMedLoader
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS,
    QUESTION_TARGETS, PUBMED_MAX_DOCS,
    QA_OUTPUT_FILE, INTERMEDIATE_DIR
)


class QAGenerator:
    """Question-Answer pair generator using KG-RAG framework"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY
        )
        self.dataset = []
        self.question_hashes = set()
        self.counts = {
            "One-hop": 0,
            "Two-hop": 0,
            "Intersection": 0,
            "Attribute": 0
        }
        self.id_counter = 0
        
    def test_pubmed_search(self):
        """Test PubMed search functionality"""
        print("\n[TEST] Testing PubMed search...")
        test_query = "cancer"
        try:
            loader = PubMedLoader(query=test_query, load_max_docs=1)
            docs = loader.load()
            if docs:
                print(f"[SUCCESS] PubMed search successful: {test_query}")
            else:
                print(f"[WARNING] No PubMed results for: {test_query}")
        except Exception as e:
            print(f"[ERROR] PubMed search error: {str(e)}")
    
    def check_pubmed_data(self, head, tail):
        """Retrieve PubMed documents for entity pair"""
        query = f"{head} {tail}"
        try:
            loader = PubMedLoader(query=query, load_max_docs=PUBMED_MAX_DOCS)
            docs = loader.load()
            if docs:
                return True, "\n".join([doc.page_content for doc in docs])
            else:
                return False, "No PubMed data retrieved"
        except Exception as e:
            print(f"[ERROR] PubMed search error: {str(e)}")
            return False, "No PubMed data retrieved"
    
    def save_intermediate_results(self, count):
        """Save intermediate results"""
        output_file = INTERMEDIATE_DIR / f"qa_pairs_intermediate_{count}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Intermediate results saved to {output_file}")
    
    def parse_llm_response(self, response_content):
        """Parse LLM response to extract question and answer"""
        lines = response_content.strip().split("\n")
        question, answer = "", ""
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("question"):
                current_section = "question"
                question = line.split(":", 1)[1].strip() if ":" in line else line[len("question"):].strip()
            elif line.lower().startswith("answer"):
                current_section = "answer"
                answer = line.split(":", 1)[1].strip() if ":" in line else line[len("answer"):].strip()
            elif current_section == "question":
                question += "\n" + line
            elif current_section == "answer":
                answer += "\n" + line
        
        return question.strip(), answer.strip()
    
    def generate_onehop_qa(self, session, target_count):
        """Generate One-hop questions"""
        print("[INFO] Generating One-hop questions...")
        rel_types = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel_type")
        rel_types = [r["rel_type"] for r in rel_types]
        
        for rel in rel_types:
            if self.counts["One-hop"] >= target_count:
                break
            triples = session.run(
                f"MATCH (h)-[r:{rel}]->(t) "
                f"RETURN h.name AS head, h.type AS head_type, type(r) AS rel, "
                f"t.name AS tail, t.type AS tail_type, labels(h) AS head_labels, "
                f"labels(t) AS tail_labels LIMIT 1000"
            )
            triples = [r for r in triples if r["head"] and r["tail"]]
            random.shuffle(triples)
            
            for record in triples:
                if self.counts["One-hop"] >= target_count:
                    break
                    
                head, rel_type, tail = record["head"], record["rel"], record["tail"]
                head_labels = record.get("head_labels", [])
                tail_labels = record.get("tail_labels", [])
                
                has_pubmed_data, context = self.check_pubmed_data(head, tail)
                if not has_pubmed_data:
                    continue
                
                prompt = self._get_onehop_prompt(head, head_labels, rel_type, tail, tail_labels, context)
                qa_item = self._generate_qa_with_retry(prompt, "One-hop", {
                    "head": head, "head_type": record.get("head_type"),
                    "relation": rel_type,
                    "tail": tail, "tail_type": record.get("tail_type"),
                    "pubmed_query": f"{head} {tail}", "pubmed_context": context
                })
                
                if qa_item:
                    self.dataset.append(qa_item)
                    self.counts["One-hop"] += 1
                    self.id_counter += 1
                    if len(self.dataset) % 50 == 0:
                        self.save_intermediate_results(len(self.dataset))
    
    def generate_twohop_qa(self, session, target_count):
        """Generate Two-hop questions"""
        print("[INFO] Generating Two-hop questions...")
        paths = session.run(
            "MATCH (h)-[r1]->(m)-[r2]->(t) "
            "RETURN h.name AS head, h.type AS head_type, type(r1) AS rel1, "
            "m.name AS mid, m.type AS mid_type, type(r2) AS rel2, "
            "t.name AS tail, t.type AS tail_type, labels(h) AS head_labels, "
            "labels(m) AS mid_labels, labels(t) AS tail_labels LIMIT 1000"
        )
        paths = [r for r in paths if r["head"] and r["mid"] and r["tail"]]
        random.shuffle(paths)
        
        for record in paths:
            if self.counts["Two-hop"] >= target_count:
                break
                
            head, rel1, mid, rel2, tail = (
                record["head"], record["rel1"], record["mid"], 
                record["rel2"], record["tail"]
            )
            head_labels = record.get("head_labels", [])
            mid_labels = record.get("mid_labels", [])
            tail_labels = record.get("tail_labels", [])
            
            has_pubmed_data, context = self.check_pubmed_data(head, tail)
            if not has_pubmed_data:
                continue
            
            prompt = self._get_twohop_prompt(
                head, head_labels, rel1, mid, mid_labels, rel2, tail, tail_labels, context
            )
            qa_item = self._generate_qa_with_retry(prompt, "Two-hop", {
                "head": head, "head_type": record.get("head_type"),
                "relation1": rel1, "mid": mid, "mid_type": record.get("mid_type"),
                "relation2": rel2, "tail": tail, "tail_type": record.get("tail_type"),
                "pubmed_query": f"{head} {tail}", "pubmed_context": context
            })
            
            if qa_item:
                self.dataset.append(qa_item)
                self.counts["Two-hop"] += 1
                self.id_counter += 1
                if len(self.dataset) % 50 == 0:
                    self.save_intermediate_results(len(self.dataset))
    
    def generate_intersection_qa(self, session, target_count):
        """Generate Intersection questions"""
        print("[INFO] Generating Intersection questions...")
        intersections = session.run("""
            MATCH (h1)-[r1]->(c)<-[r2]-(h2)
            WHERE h1 <> h2
            RETURN h1.name AS head1, h1.type AS head1_type, type(r1) AS rel1, 
            c.name AS common, c.type AS common_type, type(r2) AS rel2, 
            h2.name AS head2, h2.type AS head2_type, 
            labels(h1) AS head1_labels, labels(c) AS common_labels, 
            labels(h2) AS head2_labels
            LIMIT 1000
        """)
        intersections = [r for r in intersections if r["head1"] and r["common"] and r["head2"]]
        random.shuffle(intersections)
        
        for record in intersections:
            if self.counts["Intersection"] >= target_count:
                break
                
            head1, rel1, common, rel2, head2 = (
                record["head1"], record["rel1"], record["common"], 
                record["rel2"], record["head2"]
            )
            head1_labels = record.get("head1_labels", [])
            common_labels = record.get("common_labels", [])
            head2_labels = record.get("head2_labels", [])
            
            has_pubmed_data, context = self.check_pubmed_data(head1, head2)
            if not has_pubmed_data:
                continue
            
            prompt = self._get_intersection_prompt(
                head1, head1_labels, rel1, common, common_labels, 
                rel2, head2, head2_labels, context
            )
            qa_item = self._generate_qa_with_retry(prompt, "Intersection", {
                "head1": head1, "head1_type": record.get("head1_type"),
                "relation1": rel1, "common": common, "common_type": record.get("common_type"),
                "relation2": rel2, "head2": head2, "head2_type": record.get("head2_type"),
                "pubmed_query": f"{head1} {head2}", "pubmed_context": context
            })
            
            if qa_item:
                self.dataset.append(qa_item)
                self.counts["Intersection"] += 1
                self.id_counter += 1
                if len(self.dataset) % 50 == 0:
                    self.save_intermediate_results(len(self.dataset))
    
    def generate_attribute_qa(self, session, target_count):
        """Generate Attribute questions"""
        print("[INFO] Generating Attribute questions...")
        attributes = session.run("""
            MATCH (e) WHERE e.name IS NOT NULL AND e.description IS NOT NULL
            RETURN e.name AS entity, e.description AS description, labels(e) AS labels 
            LIMIT 10000
        """)
        attributes = [r for r in attributes if r["entity"] and r["description"]]
        random.shuffle(attributes)
        
        for record in attributes:
            if self.counts["Attribute"] >= target_count:
                break
                
            entity, description = record["entity"], record["description"]
            labels = record.get("labels", [])
            
            has_pubmed_data, context = self.check_pubmed_data(entity, description)
            if not has_pubmed_data:
                continue
            
            prompt = self._get_attribute_prompt(entity, description, labels, context)
            qa_item = self._generate_qa_with_retry(prompt, "Attribute", {
                "entity": entity, "entity_type": description,
                "labels": labels, "pubmed_query": f"{entity} {description}",
                "pubmed_context": context
            })
            
            if qa_item:
                self.dataset.append(qa_item)
                self.counts["Attribute"] += 1
                self.id_counter += 1
                if len(self.dataset) % 50 == 0:
                    self.save_intermediate_results(len(self.dataset))
    
    def _generate_qa_with_retry(self, prompt, question_type, metadata, max_retries=2):
        """Generate QA with retry logic"""
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                response = self.llm.invoke(prompt)
                question, answer = self.parse_llm_response(response.content)
                
                if question and answer:
                    success = True
                    if question in self.question_hashes:
                        print(f"[WARNING] 중복된 질문 발견. 다음 케이스로 넘어갑니다.")
                        return None
                    
                    self.question_hashes.add(question)
                    return {
                        "id": self.id_counter,
                        "question_type": question_type,
                        "question": question,
                        "answer": answer,
                        **metadata
                    }
                else:
                    retry_count += 1
                    print(f"[WARNING] LLM이 질문이나 답변을 생성하지 못함. 재시도 {retry_count}/{max_retries}")
            except Exception as e:
                retry_count += 1
                print(f"[WARNING] LLM generation failed (retry {retry_count}/{max_retries}): {str(e)}")
        
        if not success:
            print(f"[WARNING] 최대 재시도 횟수 초과. 다음 케이스로 넘어갑니다.")
        
        return None
    
    def _get_onehop_prompt(self, head, head_labels, rel, tail, tail_labels, context):
        """Get prompt for One-hop questions"""
        return (
            "You are a biomedical expert. Your task is to generate a short-answer style "
            "medical exam question and a clear answer based on the given information. "
            "IMPORTANT: Follow the exact format of the examples below.\n\n"
            "EXAMPLES (Follow these formats exactly):\n"
            "Example 1:\n"
            "Question: Which types of cancer are associated with MET?\n"
            "Answer: MET is associated with low-grade glioma, renal clear cell carcinoma, "
            "and papillary renal cell carcinoma.\n\n"
            "Example 2:\n"
            "Question: What drugs can treat cancers with TERT mutations?\n"
            "Answer: Cancers with TERT mutations can be inhibited by doxorubicin.\n\n"
            "IMPORTANT RULES:\n"
            "1. Keep questions and answers short and concise\n"
            "2. Do not include explanations or rationales\n"
            "3. Do not use multiple choice format\n"
            "4. Focus on specific disease, treatment, or molecular mechanism\n"
            "5. Make it challenging but answerable\n\n"
            f"Now, generate a question and answer based on the following information:\n"
            f"Subgraph:\n- Head entity ({head_labels[0] if head_labels else 'Unknown'}): {head}\n"
            f"- Relation: {rel}\n- Tail entity ({tail_labels[0] if tail_labels else 'Unknown'}): {tail}\n"
            f"PubMed context:\n{context}\n"
            f"Output:\nQuestion:\nAnswer:"
        )
    
    def _get_twohop_prompt(self, head, head_labels, rel1, mid, mid_labels, rel2, tail, tail_labels, context):
        """Get prompt for Two-hop questions"""
        return (
            "You are a biomedical expert. Your task is to generate a short-answer style "
            "medical exam question and a clear answer based on the given information. "
            "IMPORTANT: Follow the exact format of the examples below.\n\n"
            "EXAMPLES (Follow these formats exactly):\n"
            "Example 1:\n"
            "Question: Which gene is expressed in the brain and regulates dopamine levels?\n"
            "Answer: COMT is expressed in the brain and regulates dopamine levels.\n\n"
            "IMPORTANT RULES:\n"
            "1. Keep questions and answers short and concise\n"
            "2. Do not include explanations or rationales\n"
            "3. Do not use multiple choice format\n"
            "4. Focus on specific disease, treatment, or molecular mechanism\n"
            "5. Make it challenging but answerable\n\n"
            f"Now, generate a question and answer based on the following information:\n"
            f"Subgraph:\n- Head entity ({head_labels[0] if head_labels else 'Unknown'}): {head}\n"
            f"- Relation1: {rel1}\n- Intermediate entity ({mid_labels[0] if mid_labels else 'Unknown'}): {mid}\n"
            f"- Relation2: {rel2}\n- Tail entity ({tail_labels[0] if tail_labels else 'Unknown'}): {tail}\n"
            f"PubMed context:\n{context}\n"
            f"Output:\nQuestion:\nAnswer:"
        )
    
    def _get_intersection_prompt(self, head1, head1_labels, rel1, common, common_labels, rel2, head2, head2_labels, context):
        """Get prompt for Intersection questions"""
        return (
            "You are a biomedical expert. Your task is to generate a short-answer style "
            "medical exam question and a clear answer based on the given information. "
            "IMPORTANT: Follow the exact format of the examples below.\n\n"
            "EXAMPLES (Follow these formats exactly):\n"
            "Example 1:\n"
            "Question: What is the common target of both EGFR and HER2 inhibitors?\n"
            "Answer: PI3K is the common target of both EGFR and HER2 inhibitors.\n\n"
            "IMPORTANT RULES:\n"
            "1. Keep questions and answers short and concise\n"
            "2. Do not include explanations or rationales\n"
            "3. Do not use multiple choice format\n"
            "4. Focus on specific disease, treatment, or molecular mechanism\n"
            "5. Make it challenging but answerable\n\n"
            f"Now, generate a question and answer based on the following information:\n"
            f"Subgraph:\n- Head1 entity ({head1_labels[0] if head1_labels else 'Unknown'}): {head1}\n"
            f"- Relation1: {rel1}\n- Common entity ({common_labels[0] if common_labels else 'Unknown'}): {common}\n"
            f"- Relation2: {rel2}\n- Head2 entity ({head2_labels[0] if head2_labels else 'Unknown'}): {head2}\n"
            f"PubMed context:\n{context}\n"
            f"Output:\nQuestion:\nAnswer:"
        )
    
    def _get_attribute_prompt(self, entity, description, labels, context):
        """Get prompt for Attribute questions"""
        return (
            "You are a biomedical expert. Your task is to generate a short-answer style "
            "medical exam question and a clear answer based on the given information. "
            "IMPORTANT: Follow the exact format of the examples below.\n\n"
            "EXAMPLES (Follow these formats exactly):\n"
            "Example 1:\n"
            "Question: What is the molecular function of TP53?\n"
            "Answer: TP53 functions as a tumor suppressor protein.\n\n"
            "IMPORTANT RULES:\n"
            "1. Keep questions and answers short and concise\n"
            "2. Do not include explanations or rationales\n"
            "3. Do not use multiple choice format\n"
            "4. Focus on specific disease, treatment, or molecular mechanism\n"
            "5. Make it challenging but answerable\n\n"
            f"Now, generate a question and answer based on the following information:\n"
            f"Entity: {entity} (Description: {description})\n"
            f"Labels: {labels}\n"
            f"PubMed context:\n{context}\n"
            f"Output:\nQuestion:\nAnswer:"
        )
    
    def generate_all(self):
        """Generate all question types"""
        self.test_pubmed_search()
        
        with self.driver.session() as session: 
            # Generate each question type
            self.generate_onehop_qa(session, QUESTION_TARGETS["One-hop"])
            self.generate_twohop_qa(session, QUESTION_TARGETS["Two-hop"])
            self.generate_intersection_qa(session, QUESTION_TARGETS["Intersection"])
            self.generate_attribute_qa(session, QUESTION_TARGETS["Attribute"])
        
        # Save final results
        with open(QA_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*50)
        print("[SUCCESS] QA dataset generation completed!")
        print(f"[INFO] Generated {len(self.dataset)} QA pairs")
        print(f"[INFO] One-hop: {self.counts['One-hop']}, Two-hop: {self.counts['Two-hop']}, "
              f"Intersection: {self.counts['Intersection']}, Attribute: {self.counts['Attribute']}")
        print(f"[INFO] Results saved to {QA_OUTPUT_FILE}")
        print("="*50 + "\n")
    
    def close(self):
        """Close database connection"""
        self.driver.close()


if __name__ == "__main__":
    generator = QAGenerator()
    try:
        generator.generate_all()
    finally:
        generator.close()

