"""
GPT-based Evaluator for QA pairs
Evaluates question naturalness and answer appropriateness
"""
import json
from langchain_openai import ChatOpenAI
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import OPENAI_API_KEY, TEMPERATURE


class GPTEvaluator:
    """GPT-based evaluator for QA pairs"""
    
    def __init__(self, model="gpt-4", temperature=TEMPERATURE):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=10,
            openai_api_key=OPENAI_API_KEY
        )
    
    def get_naturalness_prompt(self, question):
        """Get prompt for naturalness evaluation"""
        return (
            "You are a biomedical expert evaluating the quality of a question. "
            "Rate the following question for naturalness (how well it reads as a natural, "
            "expert-level question) on a scale of 1 to 5 (1 being very unnatural, "
            "5 being very natural). Provide only the score as a number.\n\n"
            f"Question: {question}\nScore:"
        )
    
    def get_appropriateness_prompt(self, question, answer):
        """Get prompt for appropriateness evaluation"""
        return (
            "You are a biomedical expert evaluating the quality of an answer. "
            "Rate the following answer for appropriateness (how well it answers the question) "
            "on a scale of 1 to 5 (1 being very inappropriate, 5 being very appropriate). "
            "Provide only the score as a number.\n\n"
            f"Question: {question}\nAnswer: {answer}\nScore:"
        )
    
    def evaluate_qa(self, question, answer):
        """Evaluate a single QA pair"""
        # Naturalness evaluation
        naturalness_prompt = self.get_naturalness_prompt(question)
        naturalness_response = self.llm.invoke(naturalness_prompt)
        try:
            naturalness_score = int(naturalness_response.content.strip())
        except Exception:
            naturalness_score = 0
        
        # Appropriateness evaluation
        appropriateness_prompt = self.get_appropriateness_prompt(question, answer)
        appropriateness_response = self.llm.invoke(appropriateness_prompt)
        try:
            appropriateness_score = int(appropriateness_response.content.strip())
        except Exception:
            appropriateness_score = 0
        
        return {
            "naturalness_score": naturalness_score,
            "answer_appropriateness_score": appropriateness_score
        }
    
    def evaluate_dataset(self, qa_pairs, output_file):
        """Evaluate entire dataset"""
        scores = []
        for idx, qa in enumerate(qa_pairs):
            question = qa["question"]
            answer = qa["answer"]
            
            result = self.evaluate_qa(question, answer)
            result.update({
                "question": question,
                "answer": answer
            })
            scores.append(result)
            
            print(f"[INFO] {idx+1}/{len(qa_pairs)} evaluated - "
                  f"Naturalness: {result['naturalness_score']}, "
                  f"Appropriateness: {result['answer_appropriateness_score']}")
        
        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Evaluation completed! Results saved to {output_file}")
        return scores


if __name__ == "__main__":
    import argparse
    from config.config import QA_OUTPUT_FILE, OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description="Evaluate QA pairs using GPT")
    parser.add_argument("--input", type=str, default=str(QA_OUTPUT_FILE),
                       help="Input QA pairs file")
    parser.add_argument("--output", type=str, 
                       default=str(OUTPUT_DIR / "evaluation_results" / "scores_gpt4.json"),
                       help="Output evaluation results file")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="GPT model to use")
    
    args = parser.parse_args()
    
    # Load QA pairs
    with open(args.input, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    
    # Evaluate
    evaluator = GPTEvaluator(model=args.model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    evaluator.evaluate_dataset(qa_pairs, args.output)

