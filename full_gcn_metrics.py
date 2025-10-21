# run_paper_gcn_comprehensive.py
import os
import time
import json
import pickle
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.optim as optim

# Import project modules - EXACTLY as in paper
try:
    from eagle.predictors.gcn.gcn import GCN
    from eagle.predictors import infer
    from eagle.models import nasbench201
except ImportError as e:
    print("ERROR: Could not import project modules.")
    print("Please run this script from the root directory of the brp-nas project.")
    raise e

# -------------------- CONFIG (from YAML) --------------------
PICKLE_DATASET = "results/desktop-cpu-core-i7-7820x.pickle" #! مسیر را در صورت نیاز تغییر دهید
RESULTS_DIR = Path("results/paper_gcn_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CFG_PAPER = {
    'num_features': 6,
    'num_layers': 4,
    'num_hidden': 600,
    'dropout_ratio': 2.0e-3,
    'lr': 8.0e-4,
    'weight_decay': 5.0e-4,
    'batch_size': 10,
    'epochs': 250,
}

# Dataset config
DATA_CFG = {
    'total_points': 15284, # Using the larger dataset size from YAML
    'train_points': 900,
    'val_points': 100
}
DATA_CFG['test_points'] = DATA_CFG['total_points'] - DATA_CFG['train_points'] - DATA_CFG['val_points']


CPU_TDP_WATTS = 65.0 #! این مقدار را برای CPU خود تنظیم کنید

# -------------------- ADDED METRICS FUNCTIONS --------------------
def compute_accuracy_thresholds(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    relative_errors = np.abs((y_true - y_pred) / denom)
    return (
        np.mean(relative_errors <= 0.01) * 100,
        np.mean(relative_errors <= 0.05) * 100,
        np.mean(relative_errors <= 0.10) * 100,
        np.mean(relative_errors <= 0.20) * 100,
    )

def energy_estimate_from_cpu(seconds, avg_cpu, tdp=CPU_TDP_WATTS):
    return (avg_cpu / 100.0) * tdp * seconds

# -------------------- EXACT PAPER TRAINING (with comprehensive metrics) --------------------
def train_paper_gcn(train_data, val_data, test_data, cfg):
    print("=== TRAINING PAPER GCN (EXACT CODE + COMPREHENSIVE METRICS) ===")
    
    device = torch.device("cpu")
    model = GCN(
        num_features=cfg['num_features'],
        num_layers=cfg['num_layers'],
        num_hidden=cfg['num_hidden'],
        dropout_ratio=cfg['dropout_ratio']
    )
    model.double().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.L1Loss()
    
    # === METRICS START: Initialize monitoring variables ===
    process = psutil.Process(os.getpid())
    total_start_time = time.time()
    mem_before = process.memory_info().rss / 1024.0 / 1024.0
    cpu_samples = []
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_path = RESULTS_DIR / "paper_gcn_best.pt"
    # === METRICS END: Initialization ===

    # Training loop - EXACT PAPER LOGIC
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        for i in range(0, len(train_data), cfg['batch_size']):
            batch = train_data[i:i+cfg['batch_size']]
            graphs, targets = [item[0] for item in batch], [item[1] for item in batch]
            
            adjacency, features, latency, _ = infer.prepare_tensors(graphs, targets, nasbench201, False, False)
            adjacency, features, latency = adjacency.double().to(device), features.double().to(device), latency.double().to(device)

            optimizer.zero_grad()
            preds = model(adjacency, features)
            loss = criterion(preds, latency)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            batch_count += 1
        
        # === METRICS: Record epoch data ===
        cpu_samples.append(process.cpu_percent(interval=None))
        avg_epoch_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_epoch_train_loss)
        # === METRICS END: Epoch data ===

        # Validation logic
        model.eval()
        epoch_val_loss = 0.0
        vcount = 0
        with torch.no_grad():
            for graph, target in val_data:
                adj, feat, lat, _ = infer.prepare_tensors([graph], [target], nasbench201, False, False)
                pred = model(adj.double().to(device), feat.double().to(device))
                epoch_val_loss += criterion(pred, lat.double().to(device)).item()
                vcount += 1
        
        avg_epoch_val_loss = epoch_val_loss / vcount
        val_losses.append(avg_epoch_val_loss)

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            torch.save(model.state_dict(), best_path)

        if epoch % 50 == 0 or epoch == cfg['epochs'] - 1:
            print(f"[Epoch {epoch+1:03d}/{cfg['epochs']}] Train Loss: {avg_epoch_train_loss:.6f} | Val Loss: {avg_epoch_val_loss:.6f}")

    # === METRICS START: Finalize resource measurements ===
    total_training_time = time.time() - total_start_time
    mem_after = process.memory_info().rss / 1024.0 / 1024.0
    mem_used = mem_after - mem_before
    avg_cpu = np.mean(cpu_samples)
    total_energy = energy_estimate_from_cpu(total_training_time, avg_cpu)
    # === METRICS END: Resource measurements ===

    # Load best model and test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    preds, targets = [], []
    inference_start_time = time.time()
    with torch.no_grad():
        for graph, target in test_data:
            adj, feat, _, _ = infer.prepare_tensors([graph], [target], nasbench201, False, False)
            pred = model(adj.double().to(device), feat.double().to(device))
            preds.append(pred.item())
            targets.append(target)
    inference_time = time.time() - inference_start_time

    # === METRICS START: Calculate all final metrics ===
    mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
    r2 = 1 - (np.sum((np.array(targets) - np.array(preds))**2) / np.sum((np.array(targets) - np.mean(targets))**2))
    acc1, acc5, acc10, acc20 = compute_accuracy_thresholds(targets, preds)
    abs_errors = np.abs(np.array(preds) - np.array(targets))
    
    results = {
        'model_name': 'Paper_GCN_Baseline',
        'accuracy': {
            'mae': mae,
            'r2': r2,
            'accuracy_1%': acc1, 'accuracy_5%': acc5,
            'accuracy_10%': acc10, 'accuracy_20%': acc20
        },
        'error_analysis': {
            'min_abs_error': float(min(abs_errors)),
            'max_abs_error': float(max(abs_errors)),
            'mean_abs_error': float(np.mean(abs_errors)),
            'target_range': [min(targets), max(targets)],
            'prediction_range': [min(preds), max(preds)],
        },
        'performance': {
            'total_training_time_seconds': total_training_time,
            'inference_time_seconds': inference_time,
            'memory_usage_mb': mem_used,
            'avg_cpu_percent': avg_cpu,
            'total_energy_joules': total_energy,
        },
        'training_details': {
            'epochs_trained': cfg['epochs'],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'train_losses_per_epoch': train_losses,
            'val_losses_per_epoch': val_losses,
        },
        'model_specs': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'architecture': f"GCN_{cfg['num_layers']}L_{cfg['num_hidden']}H"
        }
    }
    # === METRICS END: Final calculation ===

    print("\n--- COMPREHENSIVE BASELINE RESULTS ---")
    print(json.dumps(results, indent=4, default=lambda x: round(x, 6)))
    return results

# -------------------- MAIN --------------------
def main():
    print("Loading dataset...")
    with open(PICKLE_DATASET, 'rb') as f:
        items = list(pickle.load(f).items())
    print(f"Total items found: {len(items)}")
    
    train_data = items[:DATA_CFG['train_points']]
    val_data = items[DATA_CFG['train_points'] : DATA_CFG['train_points'] + DATA_CFG['val_points']]
    test_data = items[DATA_CFG['train_points'] + DATA_CFG['val_points']:] # Use the rest for testing
    
    print(f"Data split -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    results = train_paper_gcn(train_data, val_data, test_data, CFG_PAPER)

    # Save results
    with open(RESULTS_DIR / "paper_gcn_comprehensive_results.json", "w") as f:
        json.dump(results, f, indent=4)
    df = pd.json_normalize(results, sep='_')
    df.to_csv(RESULTS_DIR / "paper_gcn_comprehensive_results.csv", index=False)
    print(f"\n✅ Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()