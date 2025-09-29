# run_final_baselines.py
import sys
import os
sys.path.append('.')

def main():
    """ارزیابی نهایی با GCN آموزش‌دیده"""
    try:
        print("🚀 شروع ارزیابی نهایی")
        print("=" * 50)
        
        # بارگذاری دیتاست
        dataset_path = "results/desktop-cpu-core-i7-7820x.pickle"
        print(f"📁 بارگذاری دیتاست از: {dataset_path}")
        
        import pickle
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # تبدیل به لیست
        data_pairs = list(dataset.items())[:1000]  # 1000 داده اول
        
        print(f"📊 تعداد داده‌ها: {len(data_pairs)}")
        
        # تقسیم داده
        train_data = data_pairs[:800]  # 800 آموزش
        test_data = data_pairs[800:900]  # 100 تست
        
        print(f"🔢 آموزش: {len(train_data)}, تست: {len(test_data)}")
        
        # بارگذاری GCN آموزش‌دیده
        print("🔍 بارگذاری GCN آموزش‌دیده...")
        
        import torch
        from eagle.predictors.gcn.gcn import GCN
        from pathlib import Path
        
        gcn_model = GCN(num_features=6, num_layers=4, num_hidden=600)
        
        # پیدا کردن آخرین مدل آموزش‌دیده
        model_dir = Path("results/nasbench201/latency/cpu/gcn/")
        model_files = list(model_dir.glob("predictor*.pt"))
        
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            gcn_model.load_state_dict(torch.load(latest_model))
            print(f"✅ مدل بارگذاری شد: {latest_model}")
        else:
            print("⚠️ مدل آموزش‌دیده یافت نشد!")
            return
        
        if torch.cuda.is_available():
            gcn_model.cuda()
        gcn_model.eval()
        
        # استخراج embeddings
        print("🔍 استخراج embeddings...")
        
        from eagle.models import nasbench201
        from eagle.predictors import infer
        import numpy as np
        
        def get_embeddings(data, gcn_model, model_module):
            embeddings = []
            targets = []
            
            for i, (graph, target) in enumerate(data):
                if i % 100 == 0:
                    print(f"   📊 پردازش: {i}/{len(data)}")
                
                try:
                    adjacency, features, _, _ = infer.prepare_tensors(
                        [graph], [target], model_module, False, False
                    )
                    
                    with torch.no_grad():
                        if hasattr(gcn_model, 'extract_features'):
                            emb = gcn_model.extract_features(adjacency, features)
                        else:
                            x = gcn_model.forward_single_model(adjacency, features)
                            emb = x[:, 0]
                        
                        embeddings.append(emb.cpu().numpy().flatten())
                        targets.append(target)
                        
                except Exception as e:
                    print(f"   ⚠️ خطا: {e}")
                    continue
            
            return np.array(embeddings), np.array(targets)
        
        X_train, y_train = get_embeddings(train_data, gcn_model, nasbench201)
        X_test, y_test = get_embeddings(test_data, gcn_model, nasbench201)
        
        print(f"✅ embeddings: {X_train.shape} -> {X_test.shape}")
        
        # آموزش مدل‌های پایه
        print("🎯 آموزش مدل‌های پایه...")
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        import time
        import pandas as pd
        from pathlib import Path
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(),
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   🎯 {name}...")
            
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = model.predict(X_test)
                pred_time = time.time() - start_time
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # دقت در بازه‌ها
                accuracies = {}
                for tol in [0.01, 0.05, 0.10, 0.20]:
                    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= tol) * 100
                    accuracies[f'acc_{int(tol*100)}%'] = accuracy
                
                results[name] = {
                    'mae': mae,
                    'r2_score': r2,
                    'train_time': train_time,
                    'pred_time': pred_time,
                    **accuracies
                }
                
                print(f"      ✅ MAE: {mae:.6f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"      ❌ خطا: {e}")
                continue
        
        # ذخیره نتایج
        output_dir = Path("results/final_baselines")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame([{'model': k, **v} for k, v in results.items()])
        df.to_csv(output_dir / "results.csv", index=False)
        
        # نمایش نتایج
        print("\n" + "=" * 50)
        print("🏆 نتایج نهایی")
        print("=" * 50)
        
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['mae']):
            print(f"📈 {name}:")
            print(f"   MAE: {metrics['mae']:.6f}")
            print(f"   R²: {metrics['r2_score']:.4f}")
            print(f"   دقت ±1%: {metrics['acc_1%']:.1f}%")
            print(f"   دقت ±5%: {metrics['acc_5%']:.1f}%")
            print()
        
        best_model = min(results.items(), key=lambda x: x[1]['mae'])
        print(f"🎉 بهترین مدل: {best_model[0]} (MAE: {best_model[1]['mae']:.6f})")
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 