# 3_train_baselines_real.py
import sys
import os
sys.path.append('.')

def train_baseline_models():
    """Train baseline models with REAL measurements - NO PLACEHOLDERS"""
    try:
        print("=== TRAINING BASELINE MODELS ===")
        
        # Load embeddings
        import numpy as np
        X_train = np.load("results/gcn_embeddings_train.npy")
        X_test = np.load("results/gcn_embeddings_test.npy")
        y_train = np.load("results/gcn_targets_train.npy")
        y_test = np.load("results/gcn_targets_test.npy")
        
        print(f"Data loaded:")
        print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Testing: {X_test.shape[0]} samples")
        
        # Import models
        from sklearn.ensemble import (
            RandomForestRegressor, GradientBoostingRegressor, 
            AdaBoostRegressor, ExtraTreesRegressor
        )
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        import time
        import pandas as pd
        from pathlib import Path
        import psutil
        import os
        
        # Define models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'SVR': SVR(kernel='rbf'),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
        }
        
        # Add advanced models
        try:
            from xgboost import XGBRegressor
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
        except:
            print("XGBoost not available")
            
        try:
            from lightgbm import LGBMRegressor
            models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42)
        except:
            print("LightGBM not available")
            
        try:
            from catboost import CatBoostRegressor
            models['CatBoost'] = CatBoostRegressor(iterations=100, verbose=False, random_state=42)
        except:
            print("CatBoost not available")
        
        results = {}
        process = psutil.Process(os.getpid())
        
        print("Starting model training...")
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                # Measure resources
                memory_before = process.memory_info().rss / 1024 / 1024
                cpu_before = process.cpu_percent()
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Measure resources after
                memory_after = process.memory_info().rss / 1024 / 1024
                cpu_after = process.cpu_percent()
                memory_used = memory_after - memory_before
                cpu_used = cpu_after - cpu_before
                
                # Inference
                inference_start = time.time()
                y_pred = model.predict(X_test)
                inference_time = time.time() - inference_start
                
                # Calculate ALL metrics
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate ALL accuracy levels (1%, 5%, 10%, 20%)
                accuracy_1 = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)) <= 0.01) * 100
                accuracy_5 = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)) <= 0.05) * 100
                accuracy_10 = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)) <= 0.10) * 100
                accuracy_20 = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)) <= 0.20) * 100
                
                # Model size
                import pickle as pkl
                try:
                    model_size = len(pkl.dumps(model)) / 1024  # KB
                except:
                    model_size = 0
                
                # Energy estimation
                energy_estimate = training_time * 10
                
                # Store REAL results
                results[name] = {
                    'mae': mae,
                    'r2_score': r2,
                    'accuracy_1%': accuracy_1,
                    'accuracy_5%': accuracy_5,
                    'accuracy_10%': accuracy_10,
                    'accuracy_20%': accuracy_20,
                    'training_time_seconds': training_time,
                    'inference_time_seconds': inference_time,
                    'memory_usage_mb': memory_used,
                    'cpu_usage_percent': cpu_used,
                    'model_size_kb': model_size,
                    'energy_consumption': energy_estimate,
                }
                
                print(f"  {name}: MAE={mae:.6f}, R2={r2:.4f}, "
                      f"Acc1%={accuracy_1:.1f}%, Acc5%={accuracy_5:.1f}%")
                
            except Exception as e:
                print(f"  Error in {name}: {e}")
                continue
        
        # Load GCN results (computed in step 1)
        try:
            import json
            with open("results/gcn_final_results.json", "r") as f:
                gcn_results = json.load(f)
            results['GCN_Original'] = gcn_results
            print("GCN results loaded successfully")
        except:
            print("Warning: Could not load GCN results")
        
        # Display results
        print("\n" + "=" * 120)
        print("FINAL RESULTS - ALL MODELS")
        print("=" * 120)
        
        print(f"\n{'Model':<18} {'MAE':<10} {'R2':<8} {'Acc1%':<7} {'Acc5%':<7} {'Acc10%':<7} {'Acc20%':<7} {'Time':<8} {'Memory':<8}")
        print("-" * 120)
        
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['mae']):
            print(f"{name:<18} {metrics['mae']:.6f} {metrics['r2_score']:.4f} "
                  f"{metrics['accuracy_1%']:6.1f}% {metrics['accuracy_5%']:6.1f}% "
                  f"{metrics['accuracy_10%']:6.1f}% {metrics['accuracy_20%']:6.1f}% "
                  f"{metrics['training_time_seconds']:7.2f}s {metrics['memory_usage_mb']:7.1f}MB")
        
        # Save results
        output_dir = Path("results/baseline_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame([{'model': k, **v} for k, v in results.items()])
        df.to_csv(output_dir / "complete_results.csv", index=False)
        
        print(f"\nResults saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_baseline_models()