"""
run_light_gcn_pipeline.py
-------------------------
Run Light GCN + baseline models pipeline and measure TOTAL metrics.
This includes both GCN training AND baseline model training.
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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# Import project modules
from eagle.predictors.gcn.gcn import GCN
from eagle.predictors import infer
from eagle.models import nasbench201

# -------------------- CONFIG --------------------
PICKLE_DATASET = "../../../results/desktop-cpu-core-i7-7820x.pickle"
USE_TOTAL = 1520
TRAIN_N = 900
VAL_N = 1
TEST_N = 619

RESULTS_DIR = Path("results/light_gcn_pipeline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Light GCN config (simplified architecture)
CFG_LIGHT = {
    'num_features': 6,
    'num_layers': 2,
    'num_hidden': 256,
    'dropout_ratio': 0.4,
    'lr': 0.0008,
    'weight_decay': 0.0005,
    'batch_size': 10,
    'epochs': 100,  # Reduced epochs for light version
}

CPU_TDP_WATTS = 65.0

# -------------------- UTILITIES --------------------
def load_dataset(path):
    with open(path, 'rb') as f:
        ds = pickle.load(f)
    return list(ds.items())

def split_data(items, total=USE_TOTAL, train_n=TRAIN_N, val_n=VAL_N):
    items = items[:total]
    train = items[:train_n]
    val = items[train_n:train_n+val_n]
    test = items[train_n+val_n:train_n+val_n+TEST_N]
    return train, val, test

def compute_accuracy_thresholds(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return (
        np.mean(np.abs((y_true - y_pred) / denom) <= 0.01) * 100,
        np.mean(np.abs((y_true - y_pred) / denom) <= 0.05) * 100,
        np.mean(np.abs((y_true - y_pred) / denom) <= 0.10) * 100,
        np.mean(np.abs((y_true - y_pred) / denom) <= 0.20) * 100,
    )

def energy_estimate_from_cpu(train_seconds, avg_cpu_percent, cpu_tdp_watts=CPU_TDP_WATTS):
    watts = (avg_cpu_percent / 100.0) * cpu_tdp_watts
    return watts * train_seconds

# -------------------- LIGHT GCN PIPELINE --------------------
def run_light_gcn_pipeline(train_data, val_data, test_data, cfg):
    print("=== RUNNING LIGHT GCN PIPELINE ===")
    
    device = torch.device("cpu")
    process = psutil.Process(os.getpid())
    
    # Start measuring TOTAL pipeline resources (GCN + Baselines)
    total_pipeline_start = time.time()
    mem_before_total = process.memory_info().rss / 1024.0 / 1024.0
    cpu_samples_total = []

    # Phase 1: Train Light GCN
    print("\n--- Phase 1: Training Light GCN ---")
    phase1_start = time.time()
    
    model = GCN(
        num_features=cfg['num_features'],
        num_layers=cfg['num_layers'],
        num_hidden=cfg['num_hidden'],
        dropout_ratio=cfg['dropout_ratio']
    )
    model.double().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.L1Loss()

    best_val = float('inf')
    best_path = RESULTS_DIR / "light_gcn_best.pt"

    for epoch in range(cfg['epochs']):
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

        cpu_samples_total.append(process.cpu_percent(interval=None))

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
        
        avg_val = val_loss / vcount if vcount > 0 else float('nan')
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)

        if epoch % 20 == 0 or epoch == cfg['epochs']-1:
            print(f"[LightGCN] Epoch {epoch:03d} | Val Loss: {avg_val:.6f}")

    phase1_time = time.time() - phase1_start
    print(f"Light GCN training completed in {phase1_time:.2f}s")

    # Load best model for embedding extraction
    model.load_state_dict(torch.load(best_path))
    model.eval()

    # Extract embeddings
    print("\n--- Phase 1.5: Extracting Embeddings ---")
    def extract_embeddings(data_list):
        X, y = [], []
        with torch.no_grad():
            for graph, target in data_list:
                adjacency, features, _, _ = infer.prepare_tensors([graph], [target], nasbench201, False, False)
                adjacency = adjacency.double().to(device)
                features = features.double().to(device)
                x = model.forward_single_model(adjacency, features)
                emb = x[:, 0]  # global node
                emb_np = emb.cpu().numpy().reshape(-1)
                X.append(emb_np)
                y.append(float(target))
        return np.vstack(X), np.array(y)

    X_train, y_train = extract_embeddings(train_data)
    X_test, y_test = extract_embeddings(test_data)
    print(f"Embeddings extracted - Train: {X_train.shape}, Test: {X_test.shape}")

    # Phase 2: Train Baseline Models
    print("\n--- Phase 2: Training Baseline Models ---")
    baseline_models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
    }

    baseline_results = {}
    
    for name, model in baseline_models.items():
        print(f"Training {name}...")
        model_start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        model_time = time.time() - model_start_time
        cpu_samples_total.append(process.cpu_percent(interval=None))
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        acc1, acc5, acc10, acc20 = compute_accuracy_thresholds(y_test, y_pred)
        
        baseline_results[name] = {
            'mae': mae,
            'r2': r2,
            'accuracy_1%': acc1,
            'accuracy_5%': acc5,
            'training_time_seconds': model_time
        }
        
        print(f"  {name}: MAE={mae:.6f}, R²={r2:.4f}, Time={model_time:.2f}s")

    # Calculate TOTAL pipeline metrics
    total_pipeline_time = time.time() - total_pipeline_start
    mem_after_total = process.memory_info().rss / 1024.0 / 1024.0
    mem_used_total = mem_after_total - mem_before_total
    avg_cpu_total = float(np.mean(cpu_samples_total)) if cpu_samples_total else 0.0
    total_energy = energy_estimate_from_cpu(total_pipeline_time, avg_cpu_total)

    # Find best baseline model
    best_baseline_name = min(baseline_results.keys(), 
                           key=lambda x: baseline_results[x]['mae'])
    best_baseline = baseline_results[best_baseline_name]

    # Final results for the entire pipeline
    pipeline_results = {
        'model': f'LightGCN+{best_baseline_name}',
        'pipeline_components': ['Light_GCN', best_baseline_name],
        'mae': best_baseline['mae'],
        'r2': best_baseline['r2'],
        'accuracy_1%': best_baseline['accuracy_1%'],
        'accuracy_5%': best_baseline['accuracy_5%'],
        'accuracy_10%': best_baseline['accuracy_10%'],
        'accuracy_20%': best_baseline['accuracy_20%'],
        'total_pipeline_time_seconds': total_pipeline_time,
        'gcn_training_time_seconds': phase1_time,
        'baseline_training_time_seconds': best_baseline['training_time_seconds'],
        'memory_usage_mb': mem_used_total,
        'avg_cpu_percent': avg_cpu_total,
        'total_energy_joules': total_energy,
        'light_gcn_epochs': cfg['epochs'],
        'light_gcn_architecture': f"{cfg['num_layers']}L_{cfg['num_hidden']}H",
        'all_baseline_results': baseline_results
    }

    print(f"\n=== LIGHT GCN PIPELINE RESULTS ===")
    print(f"Best Model: LightGCN+{best_baseline_name}")
    print(f"MAE: {best_baseline['mae']:.6f}")
    print(f"R²: {best_baseline['r2']:.4f}")
    print(f"Total Pipeline Time: {total_pipeline_time:.2f}s")
    print(f"Total Energy: {total_energy:.2f} J")
    print(f"GCN Time: {phase1_time:.2f}s, Baseline Time: {best_baseline['training_time_seconds']:.2f}s")

    return pipeline_results

# -------------------- MAIN --------------------
def main():
    print("Loading dataset...")
    items = load_dataset(PICKLE_DATASET)
    print(f"Total items: {len(items)}")
    
    train_data, val_data, test_data = split_data(items)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Run light GCN pipeline
    results = run_light_gcn_pipeline(train_data, val_data, test_data, CFG_LIGHT)

    # Save results
    with open(RESULTS_DIR / "light_gcn_pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save to CSV
    df = pd.DataFrame([{k: v for k, v in results.items() if k != 'all_baseline_results'}])
    df.to_csv(RESULTS_DIR / "light_gcn_pipeline_results.csv", index=False)

    print(f"\nResults saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()