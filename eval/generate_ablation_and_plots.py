import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_ablation_and_plots(ablation_csv: str = "eval/ablation.csv", output_dir: str = "results", summary_csv: str = "eval/ablation_summary.csv"):
    if not os.path.exists(ablation_csv):
        print(f"Error: {ablation_csv} not found. Please run the evaluation runner first.")
        return

    df = pd.read_csv(ablation_csv)
    n_queries = len(df)
    
    # Assume mock remote model escalation pays 30 tokens (matching clients.py mock LLM behavior)
    # and all-remote baseline pays 600 tokens average per query.
    baseline_token_avg = 600
    baseline_total_tokens = n_queries * baseline_token_avg
    escalation_token_cost = 30
    
    # Define configurations
    # Formula: Combined = 0.3 * p_easy + 0.3 * agreement + 0.4 * judge_score
    configs = {
        "Full Cascade": {
            "p_easy": df["p_easy"].values,
            "agreement": df["agreement"].values,
            "judge": df["judge_score"].values
        },
        "No Classifier": {
            "p_easy": np.full(n_queries, 0.5), # Neutral prior
            "agreement": df["agreement"].values,
            "judge": df["judge_score"].values
        },
        "No Verifier (Agreement)": {
            "p_easy": df["p_easy"].values,
            "agreement": np.full(n_queries, 0.33), # Mock minimum agreement (1/N where N=3)
            "judge": df["judge_score"].values
        },
        "No Judge Gate": {
            "p_easy": df["p_easy"].values,
            "agreement": df["agreement"].values,
            "judge": np.zeros(n_queries) # No judge contribution
        }
    }
    
    # We will sweep thresholds from 0.0 to 1.0
    thresholds = np.linspace(0.0, 1.0, 101)
    
    # To store curves for plotting
    pareto_data = {}
    
    # To store summary info at default threshold (0.61 - calibrated threshold)
    calibrated_threshold = 0.61
    summary_rows = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, inputs in configs.items():
        confidences = 0.3 * inputs["p_easy"] + 0.3 * inputs["agreement"] + 0.4 * inputs["judge"]
        
        accs = []
        token_costs = []
        
        for T in thresholds:
            # If confidence >= T, we trust local model -> correctness is df["local_correct"], token cost is 0
            # If confidence < T, we escalate -> correctness is 1.0 (remote correct), token cost is escalation_token_cost
            escalate_mask = confidences < T
            
            # Compute accuracy
            correct = np.where(escalate_mask, 1.0, df["local_correct"].values)
            acc = np.mean(correct)
            accs.append(acc)
            
            # Compute tokens
            tokens = np.where(escalate_mask, escalation_token_cost, 0)
            total_tokens = np.sum(tokens)
            token_costs.append(total_tokens)
            
        pareto_data[name] = {
            "thresholds": thresholds,
            "accuracies": accs,
            "tokens": token_costs,
            "savings": [100 * (1 - (cost / baseline_total_tokens)) for cost in token_costs]
        }
        
        # Calculate specific metrics at the calibrated threshold (or nearest threshold)
        idx_cal = np.abs(thresholds - calibrated_threshold).argmin()
        cal_acc = accs[idx_cal]
        cal_tokens = token_costs[idx_cal]
        cal_savings = 100 * (1 - (cal_tokens / baseline_total_tokens))
        
        summary_rows.append({
            "Configuration": name,
            "Accuracy": f"{100*cal_acc:.1f}%",
            "Total Tokens Paid": int(cal_tokens),
            "Token Savings %": f"{cal_savings:.1f}%"
        })
        
    # Save the summary table to CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved ablation summary to {summary_csv}")
    
    # Plotting the Pareto Curve
    plt.figure(figsize=(10, 6))
    
    # Styled plot settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.grid(True, linestyle="--", alpha=0.6)
    
    colors = {
        "Full Cascade": "#6366f1", # Indigo
        "No Classifier": "#ef4444", # Red
        "No Verifier (Agreement)": "#f59e0b", # Amber
        "No Judge Gate": "#10b981" # Emerald
    }
    
    for name, data in pareto_data.items():
        # X-axis: Token Savings (higher is better) or Average Token Cost (lower is better)
        # We'll plot Accuracy (Y) vs Token Savings % (X)
        plt.plot(data["savings"], data["accuracies"], label=name, color=colors[name], linewidth=2.5)
        
        # Add marker at calibrated threshold
        idx_cal = np.abs(data["thresholds"] - calibrated_threshold).argmin()
        plt.scatter(
            [data["savings"][idx_cal]], 
            [data["accuracies"][idx_cal]], 
            color=colors[name], 
            edgecolors='black', 
            s=80, 
            zorder=5
        )
        
    plt.title("Pareto Curve: Accuracy vs. Fireworks Token Savings", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Fireworks API Token Savings (%)", fontsize=12)
    plt.ylabel("System Output Accuracy", fontsize=12)
    plt.xlim(85, 101) # Zoom into the relevant savings area
    plt.ylim(0.5, 1.05)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
    
    # Annotate calibrated threshold point on Full Cascade
    fc_data = pareto_data["Full Cascade"]
    idx_cal = np.abs(fc_data["thresholds"] - calibrated_threshold).argmin()
    plt.annotate(
        f"Calibrated Threshold ({calibrated_threshold})\nAcc: {100*fc_data['accuracies'][idx_cal]:.1f}% | Savings: {fc_data['savings'][idx_cal]:.1f}%",
        xy=(fc_data["savings"][idx_cal], fc_data["accuracies"][idx_cal]),
        xytext=(fc_data["savings"][idx_cal] - 12, fc_data["accuracies"][idx_cal] - 0.15),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
        fontweight='bold',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pareto.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated Pareto plot at {plot_path}")

if __name__ == "__main__":
    generate_ablation_and_plots()
