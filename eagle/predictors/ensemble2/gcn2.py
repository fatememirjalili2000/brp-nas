"""
run_paper_gcn_corrected.py
--------------------------
Fixed version with proper accuracy calculation and data scaling analysis.
"""

import os
import time
import json
import pickle
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# -------------------- CONFIG --------------------
PICKLE_DATASET = "../../../results/desktop-cpu-core-i7-7820x.pickle"
USE_TOTAL = 1520
TRAIN_N = 900
VAL_N = 1
TEST_N = 619

RESULTS_DIR = Path("results/paper_gcn_corrected")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CFG_PAPER = {
    'num_features': 6,
    'num_layers': 4,
    'num_hidden': 600,
    'dropout_ratio': 0.002,
    'lr': 0.0008,
    'weight_decay': 0.0005,
    'batch_size': 10,
    'epochs': 250,
}

CPU_TDP_WATTS = 65.0

# -------------------- CORRECTED UTILITIES --------------------
def load_and_analyze_dataset(path):
    """Load dataset and analyze data distribution"""
    with open(path, 'rb') as f:
        ds = pickle.load(f)
    items = list(ds.items())
    
    # Analyze data
    targets = [item[1] for item in items]
    print(f"Dataset Analysis:")
    print(f"  Total samples: {len(targets)}")
    print(f"  Target range: [{min(targets):.6f}, {max(targets):.6f}]")
    print(f"  Target mean: {np.mean(targets):.6f}")
    print(f"  Target std: {np.std(targets):.6f}")
    
    # Check for very small values
    small_values = [t for t in targets if abs(t) < 1e-5]
    print(f"  Values < 1e-5: {len(small_values)}")
    
    return items

def split_data(items, total=USE_TOTAL, train_n=TRAIN_N, val_n=VAL_N):
    items = items[:total]
    train = items[:train_n]
    val = items[train_n:train_n+val_n]
    test = items[train_n+val_n:train_n+val_n+TEST_N]
    return train, val, test

def compute_accuracy_thresholds_corrected(y_true, y_pred):
    """
    CORRECTED accuracy calculation - same as paper
    Paper uses: percentage of predictions within x% of true values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero for very small values
    # Use absolute difference for very small true values
    abs_errors = np.abs(y_true - y_pred)
    
    # For normal cases, use relative error
    # But for very small values, this can be problematic
    relative_errors = np.zeros_like(abs_errors)
    
    for i in range(len(y_true)):
        if abs(y_true[i]) > 1e-6:  # Only use relative error for reasonably sized values
            relative_errors[i] = abs_errors[i] / abs(y_true[i])
        else:
            # For very small values, use a threshold based on typical values
            # or use absolute error with a fixed threshold
            relative_errors[i] = abs_errors[i] / 1e-6  # Normalize
    
    acc1 = np.mean(relative_errors <= 0.01) * 100
    acc5 = np.mean(relative_errors <= 0.05) * 100
    acc10 = np.mean(relative_errors <= 0.10) * 100
    acc20 = np.mean(relative_errors <= 0.20) * 100
    
    return acc1, acc5, acc10, acc20

def energy_estimate_from_cpu(train_seconds, avg_cpu_percent, cpu_tdp_watts=CPU_TDP_WATTS):
    watts = (avg_cpu_percent / 100.0) * cpu_tdp_watts
    return watts * train_seconds

# -------------------- CORRECTED TRAINING --------------------
def train_paper_gcn_corrected(train_data, val_data, test_data, cfg):
    print("=== TRAINING PAPER GCN (CORRECTED) ===")
    
    # First, analyze the training data
    train_targets = [item[1] for item in train_data]
    print(f"Training targets - Min: {min(train_targets):.6f}, Max: {max(train_targets):.6f}, Mean: {np.mean(train_targets):.6f}")
    
    device = torch.device("cpu")
    
    # Import here to avoid initial import errors
    try:
        from eagle.predictors.gcn.gcn import GCN
        from eagle.predictors import infer
        from eagle.models import nasbench201
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying alternative import...")
        # Try relative imports
        import sys
        sys.path.append('../../..')
        from eagle.predictors.gcn.gcn import GCN
        from eagle.predictors import infer
        from eagle.models import nasbench201
    
    model = GCN(
        num_features=cfg['num_features'],
        num_layers=cfg['num_layers'],
        num_hidden=cfg['num_hidden'],
        dropout_ratio=cfg['dropout_ratio']
    )
    model.double().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.L1Loss()
    process = psutil.Process(os.getpid())

    # Metrics
    mem_before = process.memory_info().rss / 1024.0 / 1024.0
    cpu_samples = []
    total_start_time = time.time()

    best_val = float('inf')
    best_path = RESULTS_DIR / "paper_gcn_best.pt"
    
    # Track training progress
    train_losses = []
    val_losses = []

    for epoch in range(cfg['epochs']):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        for i in range(0, len(train_data), cfg['batch_size']):
            batch = train_data[i:i+cfg['batch_size']]
            graphs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            
            adjacency, features, latency, _ = infer.prepare_tensors(graphs, targets, nasbench201, False, False)
            adjacency = adjacency.double().to(device)
            features = features.double().to(device)
            latency = latency.double().to(device)

            optimizer.zero_grad()
            preds = model(adjacency, features)
            loss = criterion(preds, latency)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        cpu_samples.append(process.cpu_percent(interval=None))

        # Validation
        model.eval()
        val_loss = 0.0
        vcount = 0
        with torch.no_grad():
            for graph, target in val_data:
                adjacency, features, latency, _ = infer.prepare_tensors([graph], [target], nasbench201, False, False)
                adjacency = adjacency.double().to(device)
                features = features.double().to(device)
                latency = latency.double().to(device)
                pred = model(adjacency, features)
                val_loss += float(criterion(pred, latency).item())
                vcount += 1
        
        avg_train_loss = total_loss / batch_count if batch_count > 0 else float('nan')
        avg_val_loss = val_loss / vcount if vcount > 0 else float('nan')
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), best_path)

        if epoch % 50 == 0 or epoch == cfg['epochs']-1:
            epoch_time = time.time() - epoch_start
            print(f"[Epoch {epoch:03d}] Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, Time: {epoch_time:.2f}s")

    # Total metrics
    total_training_time = time.time() - total_start_time
    mem_after = process.memory_info().rss / 1024.0 / 1024.0
    mem_used = mem_after - mem_before
    avg_cpu = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    total_energy = energy_estimate_from_cpu(total_training_time, avg_cpu)

    # Load best model and test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    # Test inference
    preds, targets = [], []
    inf_start_time = time.time()
    with torch.no_grad():
        for graph, target in test_data:
            adjacency, features, latency, _ = infer.prepare_tensors([graph], [target], nasbench201, False, False)
            adjacency = adjacency.double().to(device)
            features = features.double().to(device)
            pred = model(adjacency, features)
            preds.append(float(pred.cpu().numpy()[0][0]))
            targets.append(float(target))
    inference_time = time.time() - inf_start_time

    # Analyze predictions vs targets
    print(f"\n=== PREDICTION ANALYSIS ===")
    print(f"Targets - Min: {min(targets):.6f}, Max: {max(targets):.6f}")
    print(f"Predictions - Min: {min(preds):.6f}, Max: {max(preds):.6f}")
    
    # Calculate metrics
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    acc1, acc5, acc10, acc20 = compute_accuracy_thresholds_corrected(targets, preds)
    
    # Additional analysis
    abs_errors = np.abs(np.array(targets) - np.array(preds))
    print(f"Absolute errors - Min: {min(abs_errors):.6f}, Max: {max(abs_errors):.6f}, Mean: {np.mean(abs_errors):.6f}")

    results = {
        'mae': mae,
        'r2': r2,
        'accuracy_1%': acc1,
        'accuracy_5%': acc5,
        'accuracy_10%': acc10,
        'accuracy_20%': acc20,
        'total_training_time_seconds': total_training_time,
        'inference_time_seconds': inference_time,
        'memory_usage_mb': mem_used,
        'avg_cpu_percent': avg_cpu,
        'total_energy_joules': total_energy,
        'epochs': cfg['epochs'],
        'final_train_loss': train_losses[-1] if train_losses else float('nan'),
        'final_val_loss': val_losses[-1] if val_losses else float('nan'),
        'best_val_loss': best_val,
        'target_range': [min(targets), max(targets)],
        'prediction_range': [min(preds), max(preds)],
        'mean_absolute_error': np.mean(abs_errors),
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    print(f"\n=== FINAL RESULTS ===")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy 1%: {acc1:.2f}%")
    print(f"Accuracy 5%: {acc5:.2f}%")
    print(f"Accuracy 10%: {acc10:.2f}%")
    print(f"Accuracy 20%: {acc20:.2f}%")
    print(f"Best Val Loss: {best_val:.6f}")
    print(f"Training Time: {total_training_time:.2f}s")

    return model, results, (targets, preds)

# -------------------- MAIN --------------------
def main():
    print("Loading and analyzing dataset...")
    items = load_and_analyze_dataset(PICKLE_DATASET)
    
    train_data, val_data, test_data = split_data(items)
    print(f"Split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Train with corrected implementation
    model, results, (targets, preds) = train_paper_gcn_corrected(train_data, val_data, test_data, CFG_PAPER)

    # Save results
    with open(RESULTS_DIR / "paper_gcn_corrected_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create detailed analysis
    analysis_df = pd.DataFrame({
        'target': targets,
        'prediction': preds,
        'absolute_error': np.abs(np.array(targets) - np.array(preds))
    })
    analysis_df.to_csv(RESULTS_DIR / "detailed_predictions.csv", index=False)

    # Create loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(results.get('train_losses', []), label='Train Loss')
    plt.plot(results.get('val_losses', []), label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(RESULTS_DIR / "training_loss.png")
    plt.close()

    print(f"\nResults saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()