# compare_full_light_and_baselines.py
import os
import time
import json
import pickle
import psutil
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# -------------------- CONFIG --------------------
PICKLE_DATASET = "results/desktop-cpu-core-i7-7820x.pickle"
USE_TOTAL = 1520
TRAIN_N = 900
VAL_N = 1
TEST_N = 619

RESULTS_DIR = Path("results/compare_full_light")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Full (paper-like) config
CFG_FULL = {
    'num_features': 6,
    'num_layers': 4,
    'num_hidden': 600,
    'dropout_ratio': 0.002,
    'lr': 0.0008,
    'weight_decay': 0.0005,
    'batch_size': 10,
    'epochs': 250,
}

# Light (engineer-chosen) config
CFG_LIGHT = {
    'num_features': 6,
    'num_layers': 2,
    'num_hidden': 256,
    'dropout_ratio': 0.4,   # a bit larger to regularize
    'lr': 0.0008,
    'weight_decay': 0.0005,
    'batch_size': 10,
    'epochs': 150,
}

CPU_TDP_WATTS = 65.0  # for energy est.

# -------------------- IMPORT PROJECT GCN + infer --------------------
try:
    # run from project root so `eagle` package is importable
    from eagle.predictors.gcn.gcn import GCN
    from eagle.predictors import infer
    from eagle.models import nasbench201
except Exception as e:
    print("ERROR: cannot import eagle modules. Make sure to run from project root and eagle is importable.")
    raise

# -------------------- UTIL --------------------
def load_dataset(path):
    with open(path, 'rb') as f:
        ds = pickle.load(f)
    items = list(ds.items())
    return items

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def estimate_flops_per_sample(num_nodes, in_feat, out_feat, num_layers):
    # approximate FLOPs per forward for our GCN (per sample):
    # For each layer:
    #   - features @ W : num_nodes * (in_feat * out_feat) multiplications
    #   - adjacency @ support : num_nodes * num_nodes * out_feat multiplications
    # We'll sum per-layer with layer dims.
    total = 0
    fin = in_feat
    for i in range(num_layers):
        fout = out_feat if i > 0 else out_feat  # we approximate same hidden dim for layers (our models use nhid)
        total += num_nodes * (fin * fout)  # feature * W
        total += (num_nodes * num_nodes * fout)  # adjacency matmul
        fin = fout
    # multiply by 2 to convert mults+adds ~ 2*mults (very rough)
    return total * 2

# -------------------- TRAIN FULL GCN --------------------
def train_gcn_variant(train_data, val_data, test_data, cfg, name_tag):
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
    process = psutil.Process(os.getpid())

    # measure resources
    mem_before = process.memory_info().rss / 1024.0 / 1024.0
    cpu_samples = []
    t_start = time.time()

    best_val = float('inf')
    best_path = RESULTS_DIR / f"gcn_{name_tag}_best.pt"

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

        cpu_samples.append(process.cpu_percent(interval=None))

        # validation
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
        avg_val = val_loss / vcount if vcount>0 else float('nan')
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)

        if epoch % max(1, cfg['epochs']//5) == 0 or epoch == cfg['epochs']-1:
            print(f"[{name_tag}] epoch {epoch}/{cfg['epochs']} val_loss={avg_val:.6f}")

    total_time = time.time() - t_start
    mem_after = process.memory_info().rss / 1024.0 / 1024.0
    mem_used = mem_after - mem_before
    avg_cpu = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    energy_j = energy_estimate_from_cpu(total_time, avg_cpu)

    # load best and evaluate on test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    preds = []
    targets = []
    t_inf0 = time.time()
    with torch.no_grad():
        for graph, target in test_data:
            adjacency, features, latency, _ = infer.prepare_tensors([graph], [target], nasbench201, False, False)
            adjacency = adjacency.double().to(device)
            features = features.double().to(device)
            pred = model(adjacency, features)
            preds.append(float(pred.cpu().numpy()[0][0]))
            targets.append(float(target))
    inf_time = time.time() - t_inf0

    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    acc1, acc5, acc10, acc20 = compute_accuracy_thresholds(targets, preds)

    # embeddings extraction (global node) for full model too
    X_test_emb, y_test_emb = extract_embeddings_for_model(model, test_data)
    X_train_emb, y_train_emb = extract_embeddings_for_model(model, train_data)

    params = count_parameters(model)
    # estimate FLOPs per sample using feature shapes from first batch if possible:
    # try to get node count
    try:
        # peek at first training sample
        adjacency, features, _, _ = infer.prepare_tensors([train_data[0][0]], [train_data[0][1]], nasbench201, False, False)
        num_nodes = features.shape[1]
    except:
        num_nodes = 9  # fallback
    flops = estimate_flops_per_sample(num_nodes, cfg['num_features'], cfg['num_hidden'], cfg['num_layers'])

    results = {
        'variant': name_tag,
        'mae': mae,
        'r2': r2,
        'accuracy_1%': acc1,
        'accuracy_5%': acc5,
        'accuracy_10%': acc10,
        'accuracy_20%': acc20,
        'training_time_seconds': total_time,
        'inference_time_seconds': inf_time,
        'memory_usage_mb': mem_used,
        'avg_cpu_percent': avg_cpu,
        'model_params': params,
        'flops_per_sample_est': flops,
        'energy_joules_est': energy_j
    }
    return model, (X_train_emb, y_train_emb, X_test_emb, y_test_emb), results

# -------------------- Extract embeddings helper --------------------
def extract_embeddings_for_model(model, data_list):
    device = torch.device("cpu")
    X = []
    y = []
    model.eval()
    with torch.no_grad():
        for graph, target in data_list:
            adjacency, features, _, _ = infer.prepare_tensors([graph], [target], nasbench201, False, False)
            adjacency = adjacency.double().to(device)
            features = features.double().to(device)
            x = model.forward_single_model(adjacency, features)  # batch x nodes x nhid
            emb = x[:,0]  # global node
            emb_np = emb.cpu().numpy().reshape(-1)
            X.append(emb_np)
            y.append(float(target))
    return np.vstack(X), np.array(y)

# -------------------- Train Light and regressors (if not training full separately) --------------------
# (not used because train_gcn_variant handles both full and light and extracts embeddings)

# -------------------- Train baseline ML models on embeddings --------------------
def train_baselines_on_embeddings(X_train, y_train, X_test, y_test):
    # tuned/strong defaults (engineer choices)
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5),
        'DecisionTree': DecisionTreeRegressor(max_depth=20, random_state=42),
    }

    results = {}
    process = psutil.Process(os.getpid())
    for name, m in models.items():
        t0 = time.time()
        mem_before = process.memory_info().rss / 1024.0 / 1024.0
        try:
            m.fit(X_train, y_train)
            t_train = time.time() - t0
            mem_after = process.memory_info().rss / 1024.0 / 1024.0
            mem_used = mem_after - mem_before
            y_pred = m.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            acc1, acc5, acc10, acc20 = compute_accuracy_thresholds(y_test, y_pred)
            results[name] = {
                'mae': mae, 'r2': r2,
                'accuracy_1%': acc1, 'accuracy_5%': acc5, 'accuracy_10%': acc10, 'accuracy_20%': acc20,
                'training_time_seconds': t_train, 'memory_usage_mb': mem_used
            }
            print(f"[Baselines] {name}: MAE={mae:.6f}, R2={r2:.4f}, time={t_train:.2f}s")
        except Exception as e:
            print(f"[Baselines] {name} ERROR: {e}")
            results[name] = {'error': str(e)}
    return results

# -------------------- MAIN --------------------
def main():
    try:
        items = load_dataset(PICKLE_DATASET)
    except Exception as e:
        print("Failed to load dataset:", e)
        traceback.print_exc()
        return

    print("Total items:", len(items))
    train_data, val_data, test_data = split_data(items)
    print("Split sizes:", len(train_data), len(val_data), len(test_data))

    # Full
    print("\n--- Running FULL GCN ---")
    model_full, emb_full, res_full = train_gcn_variant(train_data, val_data, test_data, {**CFG_FULL}, "full")
    Xtr_f, ytr_f, Xte_f, yte_f = emb_full

    # Light
    print("\n--- Running LIGHT GCN ---")
    model_light, emb_light, res_light = train_gcn_variant(train_data, val_data, test_data, {**CFG_LIGHT}, "light")
    Xtr_l, ytr_l, Xte_l, yte_l = emb_light

    # compute reductions (params and flops and training time)
    param_reduction = (res_full['model_params'] - res_light['model_params']) / res_full['model_params'] * 100.0
    flops_reduction = (res_full['flops_per_sample_est'] - res_light['flops_per_sample_est']) / res_full['flops_per_sample_est'] * 100.0
    time_reduction = (res_full['training_time_seconds'] - res_light['training_time_seconds']) / res_full['training_time_seconds'] * 100.0

    summary = {
        'full': res_full,
        'light': res_light,
        'param_reduction_percent': param_reduction,
        'flops_reduction_percent': flops_reduction,
        'time_reduction_percent': time_reduction,
    }

    # train baseline models on full-emb and light-emb (we'll use embeddings from light and full separately and compare)
    print("\n--- Baselines on FULL embeddings ---")
    baselines_full = train_baselines_on_embeddings(Xtr_f, ytr_f, Xte_f, yte_f)
    print("\n--- Baselines on LIGHT embeddings ---")
    baselines_light = train_baselines_on_embeddings(Xtr_l, ytr_l, Xte_l, yte_l)

    # Save everything
    out = {'summary': summary, 'baselines_full': baselines_full, 'baselines_light': baselines_light}
    with open(RESULTS_DIR / "compare_full_light_results.json", "w", encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # CSV summary (best per embedding)
    rows = []
    # full gcn baseline
    rows.append({
        'variant': 'GCN_Full',
        'mae': res_full['mae'],
        'r2': res_full['r2'],
        'time_train_s': res_full['training_time_seconds'],
        'params': res_full['model_params'],
        'flops_per_sample': res_full['flops_per_sample_est']
    })
    rows.append({
        'variant': 'GCN_Light',
        'mae': res_light['mae'],
        'r2': res_light['r2'],
        'time_train_s': res_light['training_time_seconds'],
        'params': res_light['model_params'],
        'flops_per_sample': res_light['flops_per_sample_est']
    })

    # add best baseline result for each embedding type (by MAE)
    def best_baseline(results_dict):
        best = None
        best_m = 1e9
        for k, v in results_dict.items():
            if 'mae' in v and v['mae'] < best_m:
                best = (k, v)
                best_m = v['mae']
        return best

    bfull = best_baseline(baselines_full)
    blight = best_baseline(baselines_light)
    if bfull:
        rows.append({'variant': f'FullEmbedding+{bfull[0]}', 'mae': bfull[1]['mae'], 'r2': bfull[1]['r2'], 'time_train_s': bfull[1]['training_time_seconds']})
    if blight:
        rows.append({'variant': f'LightEmbedding+{blight[0]}', 'mae': blight[1]['mae'], 'r2': blight[1]['r2'], 'time_train_s': blight[1]['training_time_seconds']})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "compare_full_light_summary.csv", index=False)

    # Plots: MAE and training time
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].bar(df['variant'], df['mae'])
    axs[0].set_title('MAE (lower better)')
    axs[1].bar(df['variant'], df['time_train_s'])
    axs[1].set_title('Train Time (s)')
    for ax in axs:
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "compare_full_light_plots.png")
    plt.close()

    # Print reductions
    print("\n=== SUMMARY ===")
    print(f"Params Full: {res_full['model_params']}, Light: {res_light['model_params']}, reduction: {param_reduction:.2f}%")
    print(f"FLOPs/sample estimate Full: {res_full['flops_per_sample_est']}, Light: {res_light['flops_per_sample_est']}, reduction: {flops_reduction:.2f}%")
    print(f"Training time reduction: {time_reduction:.2f}% (full -> light)")

    print("Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()
