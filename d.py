# 4_complete_analysis_real.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import sys
import os

def create_complete_analysis():
    """Create complete analysis with ALL accuracy levels"""
    
    # Load results
    try:
        results_df = pd.read_csv("results/baseline_comparison/complete_results.csv")
        
        # Convert to dictionary
        results = {}
        for _, row in results_df.iterrows():
            results[row['model']] = {
                'mae': row['mae'],
                'r2_score': row['r2_score'],
                'accuracy_1%': row['accuracy_1%'],
                'accuracy_5%': row['accuracy_5%'],
                'accuracy_10%': row['accuracy_10%'],
                'accuracy_20%': row['accuracy_20%'],
                'training_time_seconds': row['training_time_seconds'],
                'memory_usage_mb': row['memory_usage_mb'],
                'cpu_usage_percent': row['cpu_usage_percent'],
                'energy_consumption': row['energy_consumption']
            }
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Create output directory
    output_dir = Path("results/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("Creating comprehensive analysis...")
    
    # Sort models by MAE
    models_sorted = sorted([(k, v) for k, v in results.items()], key=lambda x: x[1]['mae'])
    model_names = [m[0] for m in models_sorted]
    
    # 1. MAE Comparison
    plt.figure(figsize=(14, 8))
    mae_values = [m[1]['mae'] for m in models_sorted]
    
    bars = plt.bar(model_names, mae_values, color='lightblue', edgecolor='darkblue')
    plt.title('MAE Comparison: All Models', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001, 
                f'{value:.6f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_mae_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy 1% Comparison
    plt.figure(figsize=(14, 8))
    acc1_values = [m[1]['accuracy_1%'] for m in models_sorted]
    
    bars = plt.bar(model_names, acc1_values, color='lightgreen', edgecolor='darkgreen')
    plt.title('Accuracy ±1% Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy ±1% (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, acc1_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_accuracy_1p_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy 5% Comparison
    plt.figure(figsize=(14, 8))
    acc5_values = [m[1]['accuracy_5%'] for m in models_sorted]
    
    bars = plt.bar(model_names, acc5_values, color='lightcoral', edgecolor='darkred')
    plt.title('Accuracy ±5% Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy ±5% (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, acc5_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_accuracy_5p_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Accuracy 10% Comparison
    plt.figure(figsize=(14, 8))
    acc10_values = [m[1]['accuracy_10%'] for m in models_sorted]
    
    bars = plt.bar(model_names, acc10_values, color='gold', edgecolor='darkorange')
    plt.title('Accuracy ±10% Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy ±10% (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, acc10_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_accuracy_10p_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Accuracy 20% Comparison
    plt.figure(figsize=(14, 8))
    acc20_values = [m[1]['accuracy_20%'] for m in models_sorted]
    
    bars = plt.bar(model_names, acc20_values, color='plum', edgecolor='purple')
    plt.title('Accuracy ±20% Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy ±20% (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, acc20_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_accuracy_20p_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. All Accuracies Combined
    plt.figure(figsize=(16, 10))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    plt.bar(x - 1.5*width, acc1_values, width, label='±1%', color='lightgreen')
    plt.bar(x - 0.5*width, acc5_values, width, label='±5%', color='lightcoral')
    plt.bar(x + 0.5*width, acc10_values, width, label='±10%', color='gold')
    plt.bar(x + 1.5*width, acc20_values, width, label='±20%', color='plum')
    
    plt.title('All Accuracy Levels Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '6_all_accuracies_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Training Time Comparison
    plt.figure(figsize=(14, 8))
    time_values = [m[1]['training_time_seconds'] for m in models_sorted]
    
    bars = plt.bar(model_names, time_values, color='orange', edgecolor='darkorange')
    plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, time_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{value:.1f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '7_training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Memory Usage Comparison
    plt.figure(figsize=(14, 8))
    memory_values = [m[1]['memory_usage_mb'] for m in models_sorted]
    
    bars = plt.bar(model_names, memory_values, color='purple', edgecolor='darkviolet', alpha=0.7)
    plt.title('Memory Usage Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, memory_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{value:.1f}MB', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '8_memory_usage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    best_model = models_sorted[0]
    gcn_model = next((m for m in models_sorted if m[0] == 'GCN_Original'), None)
    
    if gcn_model:
        improvement = ((gcn_model[1]['mae'] - best_model[1]['mae']) / gcn_model[1]['mae']) * 100
    
    report = f"""
COMPREHENSIVE ANALYSIS REPORT
=============================

Dataset Information:
- Total samples: 1520 (matching original paper)
- Training samples: 900
- Test samples: 620
- Embedding dimension: 600

Performance Summary:
-------------------
Best Model: {best_model[0]}
- MAE: {best_model[1]['mae']:.6f}
- R2 Score: {best_model[1]['r2_score']:.4f}
- Accuracy ±1%: {best_model[1]['accuracy_1%']:.1f}%
- Accuracy ±5%: {best_model[1]['accuracy_5%']:.1f}%
- Accuracy ±10%: {best_model[1]['accuracy_10%']:.1f}%
- Accuracy ±20%: {best_model[1]['accuracy_20%']:.1f}%

GCN Original Comparison:
- MAE: {gcn_model[1]['mae']:.6f} if gcn_model else 'N/A'
- Improvement: {improvement:.1f}% if gcn_model else 'N/A'

Generated Charts:
----------------
1. MAE Comparison
2. Accuracy ±1% Comparison
3. Accuracy ±5% Comparison
4. Accuracy ±10% Comparison
5. Accuracy ±20% Comparison
6. All Accuracies Combined
7. Training Time Comparison
8. Memory Usage Comparison

Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report)
    
    print("Analysis completed successfully!")
    print(f"All charts saved to: {output_dir}")

if __name__ == "__main__":
    create_complete_analysis()