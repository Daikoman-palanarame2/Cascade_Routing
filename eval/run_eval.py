import asyncio
import pandas as pd
import numpy as np
import argparse
import os
import sys

# Add src to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router.pipeline import RoutingPipeline

async def evaluate_dev_set(dev_set_path: str, output_path: str):
    print(f"Loading dev set from {dev_set_path}...")
    df = pd.read_csv(dev_set_path)
    
    pipeline = RoutingPipeline.from_env()
    
    results = []
    
    print("Running evaluation on dev set tasks...")
    for idx, row in df.iterrows():
        query = row["query"]
        local_correct = row["local_correct"]
        print(f"[{idx+1}/{len(df)}] Query: {query[:40]}...")
        
        # Run through the pipeline
        try:
            result = await pipeline.solve(query, f"eval_{idx}")
            
            # Record outcome details
            results.append({
                "query": query,
                "tier_used": result.tier,
                "tokens_paid": result.tokens_paid,
                "confidence": result.confidence,
                "local_correct": local_correct,
                "stage": result.trace.get("stage", ""),
                "p_easy": result.trace.get("p_easy", 0.5),
                "agreement": result.trace.get("agreement", 0.0),
                "judge_score": result.trace.get("judge", 0.0)
            })
        except Exception as e:
            print(f"Error solving query: {e}")
            results.append({
                "query": query,
                "tier_used": "error",
                "tokens_paid": 0,
                "confidence": 0.0,
                "local_correct": local_correct,
                "stage": "error",
                "p_easy": 0.5,
                "agreement": 0.0,
                "judge_score": 0.0
            })
            
    res_df = pd.DataFrame(results)
    
    # Save outcomes to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")
    
    # Print statistics summary
    total_queries = len(res_df)
    cache_hits = len(res_df[res_df["tier_used"] == "cache"])
    local_pass = len(res_df[res_df["tier_used"] == "local"])
    escalated = len(res_df[res_df["tier_used"] == "escalated"])
    total_tokens = res_df["tokens_paid"].sum()
    
    # Assume Fireworks all-remote baseline uses ~600 tokens average per query (input+output)
    baseline_tokens = total_queries * 600
    token_savings = 100 * (1 - (total_tokens / baseline_tokens)) if baseline_tokens > 0 else 0
    
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Total queries evaluated:  {total_queries}")
    print(f"Semantic Cache Hits:      {cache_hits} ({100*cache_hits/total_queries:.1f}%)")
    print(f"Local Model Pass:         {local_pass} ({100*local_pass/total_queries:.1f}%)")
    print(f"Escalated to 27B:         {escalated} ({100*escalated/total_queries:.1f}%)")
    print(f"Total Fireworks Tokens:   {total_tokens} (Baseline: {baseline_tokens})")
    print(f"Fireworks Token Savings:  {token_savings:.1f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cascade Routing Agent on dev set")
    parser.add_argument("--dev-set", type=str, default="eval/dev_set.csv", help="Path to dev set CSV")
    parser.add_argument("--output", type=str, default="eval/ablation.csv", help="Path to output results CSV")
    args = parser.parse_args()
    
    asyncio.run(evaluate_dev_set(args.dev_set, args.output))
