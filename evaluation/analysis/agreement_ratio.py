"""
Calculate agreement ratio between multiple evaluation models
"""
import json
import argparse
from typing import Dict, List, Any
from pathlib import Path


def load_evaluation_results(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load evaluation results - use question+answer as key"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to dictionary with question+answer as key
    result_dict = {}
    for item in data:
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()
        key = f"{question}|{answer}"
        result_dict[key] = item
    
    return result_dict


def calculate_agreement_ratio(scores1: List[int], scores2: List[int], scores3: List[int]) -> float:
    """Calculate 3-way agreement ratio"""
    if len(scores1) != len(scores2) or len(scores2) != len(scores3):
        print(f"Warning: Score list lengths differ. {len(scores1)}, {len(scores2)}, {len(scores3)}")
        return 0.0
    
    agreements = 0
    total_items = len(scores1)
    
    for i in range(total_items):
        # All three scores match
        if scores1[i] == scores2[i] == scores3[i]:
            agreements += 1
        # Two scores match (3 combinations)
        elif scores1[i] == scores2[i] or scores1[i] == scores3[i] or scores2[i] == scores3[i]:
            agreements += 0.5  # Partial match
    
    return agreements / total_items


def calculate_pairwise_agreement(scores1: List[int], scores2: List[int]) -> float:
    """Calculate pairwise agreement between two models"""
    if len(scores1) != len(scores2):
        return 0.0
    
    agreements = sum(1 for i in range(len(scores1)) if scores1[i] == scores2[i])
    return agreements / len(scores1)


def main():
    parser = argparse.ArgumentParser(description="Calculate agreement ratio between evaluation models")
    parser.add_argument("--files", nargs=3, required=True,
                       help="Three evaluation result files (JSON)")
    parser.add_argument("--names", nargs=3, required=True,
                       help="Names for the three models")
    parser.add_argument("--score-types", nargs="+", 
                       default=['naturalness_score', 'answer_appropriateness_score'],
                       help="Score types to analyze")
    
    args = parser.parse_args()
    
    # Load evaluation results
    results = {}
    for model_name, file_path in zip(args.names, args.files):
        try:
            results[model_name] = load_evaluation_results(file_path)
            print(f"[INFO] {model_name} results loaded: {len(results[model_name])} items")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name} file - {e}")
            return
    
    # Find common QA pairs
    common_keys = set(results[args.names[0]].keys())
    for model_name in args.names[1:]:
        common_keys = common_keys.intersection(set(results[model_name].keys()))
    
    print(f"\n[INFO] Common QA pairs: {len(common_keys)}")
    
    if len(common_keys) == 0:
        print("[ERROR] No common QA pairs found.")
        return
    
    print("\n" + "="*80)
    print("Agreement Ratio Analysis Results")
    print("="*80)
    
    for score_type in args.score_types:
        # Check if score type exists
        if score_type not in results[args.names[0]]:
            print(f"[WARNING] Score type '{score_type}' not found. Skipping...")
            continue
        
        print(f"\n{score_type.upper()} Analysis:")
        print("-" * 50)
        
        # Extract scores for each model
        scores = {}
        for model_name in args.names:
            scores[model_name] = [results[model_name][key].get(score_type, 0) 
                                 for key in common_keys]
        
        # 3-way agreement ratio
        three_way_agreement = calculate_agreement_ratio(
            scores[args.names[0]], 
            scores[args.names[1]], 
            scores[args.names[2]]
        )
        print(f"3-way Agreement Ratio: {three_way_agreement:.4f} ({three_way_agreement*100:.2f}%)")
        
        # Pairwise agreement ratios
        pairwise_agreements = {}
        for i in range(len(args.names)):
            for j in range(i+1, len(args.names)):
                pair_name = f"{args.names[i]}-{args.names[j]}"
                pairwise_agreements[pair_name] = calculate_pairwise_agreement(
                    scores[args.names[i]], 
                    scores[args.names[j]]
                )
                print(f"{pair_name} Agreement: {pairwise_agreements[pair_name]:.4f} "
                      f"({pairwise_agreements[pair_name]*100:.2f}%)")
        
        # Average pairwise agreement
        avg_pairwise = sum(pairwise_agreements.values()) / len(pairwise_agreements)
        print(f"Average Pairwise Agreement: {avg_pairwise:.4f} ({avg_pairwise*100:.2f}%)")
    
    print("\n" + "="*80)
    print("\nSummary:")
    print(f"Analyzed QA pairs: {len(common_keys)}")
    print(f"Analyzed models: {len(args.names)} ({', '.join(args.names)})")
    print("="*80)


if __name__ == "__main__":
    main()

