"""
run_paper_gcn_exact.py
----------------------
Run the EXACT paper GCN code but with additional metrics measurement.
No changes to the training logic, only adding measurements.
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

# Import project modules - EXACTLY as in paper
from eagle.predictors.gcn.gcn import GCN
from eagle.predictors import infer
from eagle.models import nasbench201

# -------------------- CONFIG --------------------
PICKLE_DATASET = "../../../results/desktop-cpu-core-i7-7820x.pickle"
USE_TOTAL = 1520
TRAIN_N = 900
VAL_N = 1
TEST_N = 619

RESULTS_DIR = Path("results/paper_gcn_exact")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Paper GCN config (EXACT from paper)
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

# -------------------- ADDED METRICS FUNCTIONS --------------------
def compute_accuracy_thresholds(y_true, y_pred):
    """Calculate accuracy at different thresholds"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    relative_errors = np.abs((y_true - y_pred) / denom)
    
    acc1 = np.mean(relative_errors <= 0.01) * 100
    acc5 = np.mean(relative_errors <= 0.05) * 100
    acc10 = np.mean(relative_errors <= 0.10) * 100
    acc20 = np.mean(relative_errors <= 0.20) * 100
    
    return acc1, acc5, acc10, acc20

def energy_estimate_from_cpu(train_seconds, avg_cpu_percent, cpu_tdp_watts=CPU_TDP_WATTS):
    """Estimate energy consumption in Joules"""
    watts = (avg_cpu_percent / 100.0) * cpu_tdp_watts
    return watts * train_seconds

def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_kb(model):
    """Calculate model size in KB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024.0  # Convert to KB
    return size_all_mb

# -------------------- EXACT PAPER TRAINING (with metrics) --------------------
def train_paper_gcn_with_metrics(train_data, val_data, test_data, cfg):
    """
    Exact paper training code with added metrics measurement
    """
    print("=== TRAINING PAPER GCN (EXACT CODE + METRICS) ===")
    
    device = torch.device("cpu")
    
    # Initialize model EXACTLY as in paper
    model = GCN(
        num_features=cfg['num_features'],
        num_layers=cfg['num_layers'],
        num_hidden=cfg['num_hidden'],
        dropout_ratio=cfg['dropout_ratio']
    )
    model.double().to(device)

    # Optimizer EXACTLY as in paper
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.L1Loss()

    # ========== ADDED METRICS MEASUREMENT ==========
    process = psutil.Process(os.getpid())
    
    # Start comprehensive metrics collection
    total_start_time = time.time()
    mem_before = process.memory_info().rss / 1024.0 / 1024.0  # MB
    cpu_samples = []
    # ===============================================

    best_val = float('inf')
    best_path = RESULTS_DIR / "paper_gcn_best.pt"

    # Training loop - EXACT PAPER LOGIC
    for epoch in range(cfg['epochs']):
        epoch_start = time.time()
        
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        # Exact training logic from paper
        for i in range(0, len(train_data), cfg['batch_size']):
            batch = train_data[i:i+cfg['batch_size']]
            graphs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            
            # Exact tensor preparation from paper
            adjacency, features, latency, _ = infer.prepare_tensors(graphs, targets, nasbench201, False, False)
            adjacency = adjacency.double().to(device)
            features = features.double().to(device)
            latency = latency.double().to(device)

            # Exact forward/backward from paper
            optimizer.zero_grad()
            preds = model(adjacency, features)
            loss = criterion(preds, latency)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        # ========== ADDED: CPU monitoring ==========
        cpu_samples.append(process.cpu_percent(interval=None))
        # ===========================================

        # Exact validation logic from paper
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
        
        avg_val = val_loss / vcount if vcount > 0 else float('nan')
        
        # Exact model saving logic from paper
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)

        # Progress reporting (similar to paper)
        if epoch % 50 == 0 or epoch == cfg['epochs']-1:
            epoch_time = time.time() - epoch_start
            print(f"[PaperGCN] Epoch {epoch:03d} | "
                  f"Train Loss: {total_loss/batch_count:.6f} | "
                  f"Val Loss: {avg_val:.6f} | "
                  f"Epoch Time: {epoch_time:.2f}s")

    # ========== ADDED: COMPREHENSIVE METRICS CALCULATION ==========
    total_training_time = time.time() - total_start_time
    mem_after = process.memory_info().rss / 1024.0 / 1024.0
    mem_used = mem_after - mem_before
    avg_cpu = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    total_energy = energy_estimate_from_cpu(total_training_time, avg_cpu)
    # ==============================================================

    # Load best model for testing (EXACT paper logic)
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    # Exact inference logic from paper
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
    
    # ========== ADDED: Inference time measurement ==========
    inference_time = time.time() - inf_start_time
    # =======================================================

    # Calculate paper metrics (MAE, R²)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    
    # ========== ADDED: Additional accuracy metrics ==========
    acc1, acc5, acc10, acc20 = compute_accuracy_thresholds(targets, preds)
    model_params = count_parameters(model)
    model_size_kb = get_model_size_kb(model)
    # ========================================================

    # Comprehensive results
    results = {
        # Paper metrics
        'mae': mae,
        'r2': r2,
        
        # Added accuracy metrics
        'accuracy_1%': acc1,
        'accuracy_5%': acc5,
        'accuracy_10%': acc10,
        'accuracy_20%': acc20,
        
        # Resource metrics
        'total_training_time_seconds': total_training_time,
        'inference_time_seconds': inference_time,
        'memory_usage_mb': mem_used,
        'avg_cpu_percent': avg_cpu,
        'total_energy_joules': total_energy,
        
        # Model info
        'model_parameters': model_params,
        'model_size_kb': model_size_kb,
        'epochs': cfg['epochs'],
        'architecture': f"GCN_{cfg['num_layers']}L_{cfg['num_hidden']}H",
        
        # Training config
        'batch_size': cfg['batch_size'],
        'learning_rate': cfg['lr'],
        'weight_decay': cfg['weight_decay']
    }

    print(f"\n=== PAPER GCN RESULTS (WITH COMPREHENSIVE METRICS) ===")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy 5%: {acc5:.2f}%")
    print(f"Training Time: {total_training_time:.2f}s")
    print(f"Energy: {total_energy:.2f} J")
    print(f"Memory: {mem_used:.2f} MB")
    print(f"Parameters: {model_params}")

    return model, results

# -------------------- DATA LOADING (EXACT) --------------------
def load_dataset(path):
    """Exact dataset loading from paper"""
    with open(path, 'rb') as f:
        ds = pickle.load(f)
    return list(ds.items())

def split_data(items, total=USE_TOTAL, train_n=TRAIN_N, val_n=VAL_N):
    """Exact data splitting from paper"""
    items = items[:total]
    train = items[:train_n]
    val = items[train_n:train_n+val_n]
    test = items[train_n+val_n:train_n+val_n+TEST_N]
    return train, val, test

# -------------------- MAIN --------------------
def main():
    print("Loading dataset...")
    items = load_dataset(PICKLE_DATASET)
    print(f"Total items: {len(items)}")
    
    train_data, val_data, test_data = split_data(items)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Train paper GCN with exact code but comprehensive metrics
    model, results = train_paper_gcn_with_metrics(train_data, val_data, test_data, CFG_PAPER)

    # Save comprehensive results
    with open(RESULTS_DIR / "paper_gcn_comprehensive_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save to CSV for easy comparison
    df = pd.DataFrame([results])
    df.to_csv(RESULTS_DIR / "paper_gcn_comprehensive_results.csv", index=False)

    # Create a simple results summary
    print(f"\n=== RESULTS SAVED ===")
    print(f"Location: {RESULTS_DIR}")
    print(f"Files created:")
    print(f"  - paper_gcn_comprehensive_results.json")
    print(f"  - paper_gcn_comprehensive_results.csv")
    
    # Save a quick summary text file
    with open(RESULTS_DIR / "results_summary.txt", "w") as f:
        f.write("PAPER GCN COMPREHENSIVE METRICS\n")
        f.write("================================\n\n")
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()