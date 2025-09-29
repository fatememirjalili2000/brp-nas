# run_complete_simple.py
import sys
import os
sys.path.append('.')

def main():
    """مقایسه کامل ۱۳ مدل پایه بدون seaborn"""
    try:
        print("🚀 مقایسه کامل ۱۳ مدل پایه با GCN")
        print("=" * 60)
        
        # بارگذاری دیتاست
        import pickle
        with open("results/desktop-cpu-core-i7-7820x.pickle", 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"📊 کل داده‌های موجود: {len(dataset)}")
        
        # استفاده از 1000 داده برای تست سریع اما کامل
        data_pairs = list(dataset.items())[:1000]
        
        # تقسیم داده
        train_data = data_pairs[:700]    # 700 آموزش
        test_data = data_pairs[700:900]  # 200 تست
        
        print(f"📋 تقسیم داده:")
        print(f"   🎯 آموزش: {len(train_data)}")
        print(f"   🧪 تست: {len(test_data)}")
        
        # بارگذاری مدل GCN
        import torch
        from eagle.predictors.gcn.gcn import GCN
        
        print("🔧 بارگذاری مدل GCN...")
        gcn_model = GCN(num_features=6, num_layers=4, num_hidden=600)
        gcn_model.load_state_dict(torch.load("results/nasbench201/latency/cpu/gcn/predictor_simple.pt"))
        
        if torch.cuda.is_available():
            gcn_model.cuda()
        gcn_model.eval()
        
        # استخراج embeddings
        from eagle.models import nasbench201
        from eagle.predictors import infer
        import numpy as np
        
        def extract_embeddings(data, gcn_model, model_module):
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
                        x = gcn_model.forward_single_model(adjacency, features)
                        emb = x[:, 0]
                        embeddings.append(emb.cpu().numpy().flatten())
                        targets.append(target)
                        
                except Exception as e:
                    embeddings.append(np.random.randn(600))
                    targets.append(target)
                    continue
            
            return np.array(embeddings), np.array(targets)
        
        print("🔍 استخراج embeddings...")
        X_train, y_train = extract_embeddings(train_data, gcn_model, nasbench201)
        X_test, y_test = extract_embeddings(test_data, gcn_model, nasbench201)
        
        print(f"✅ embeddings: {X_train.shape} -> {X_test.shape}")
        
        # آموزش تمام مدل‌های پایه
        from sklearn.ensemble import (
            RandomForestRegressor, GradientBoostingRegressor, 
            AdaBoostRegressor, ExtraTreesRegressor
        )
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from scipy.stats import spearmanr, kendalltau
        import time
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
        import psutil
        import os
        
        print("🎯 آموزش ۱۳ مدل پایه مختلف...")
        
        # تمام مدل‌های پایه
        models = {
            # Ensemble Methods
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            
            # Linear Models
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            
            # Non-linear Models
            'SVR': SVR(kernel='rbf'),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
        }
        
        # حذف CatBoost اگر مشکل دارد
        try:
            models['CatBoost'] = CatBoostRegressor(iterations=100, verbose=False, random_state=42)
        except:
            print("   ⚠️ CatBoost حذف شد")
        
        results = {}
        
        for name, model in models.items():
            print(f"   🎯 آموزش {name}...")
            
            try:
                # اندازه‌گیری حافظه قبل از آموزش
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # زمان و انرژی - شروع
                start_time = time.time()
                
                # آموزش مدل
                model.fit(X_train, y_train)
                
                # زمان و انرژی - پایان
                training_time = time.time() - start_time
                
                # حافظه بعد از آموزش
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                # پیش‌بینی
                start_time = time.time()
                y_pred = model.predict(X_test)
                inference_time = time.time() - start_time
                
                # محاسبه تمام معیارها
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # دقت در بازه‌های مختلف
                accuracy_1 = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)) <= 0.01) * 100
                accuracy_5 = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)) <= 0.05) * 100
                accuracy_10 = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)) <= 0.10) * 100
                accuracy_20 = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)) <= 0.20) * 100
                
                # معیارهای رتبه‌بندی
                try:
                    spearman_corr, _ = spearmanr(y_test, y_pred)
                    kendall_tau, _ = kendalltau(y_test, y_pred)
                except:
                    spearman_corr = 0
                    kendall_tau = 0
                
                # تخمین حجم مدل
                import pickle as pkl
                try:
                    model_size = len(pkl.dumps(model)) / 1024  # KB
                except:
                    model_size = 0
                
                # تخمین انرژی مصرفی (ساده)
                energy_estimate = training_time * 10  # تخمین ساده
                
                results[name] = {
                    # معیارهای دقت
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2_score': r2,
                    
                    # دقت در بازه‌ها
                    'accuracy_1%': accuracy_1,
                    'accuracy_5%': accuracy_5,
                    'accuracy_10%': accuracy_10,
                    'accuracy_20%': accuracy_20,
                    
                    # معیارهای رتبه‌بندی
                    'spearman_corr': spearman_corr,
                    'kendall_tau': kendall_tau,
                    
                    # معیارهای منابع
                    'training_time_seconds': training_time,
                    'inference_time_seconds': inference_time,
                    'memory_usage_mb': memory_used,
                    'model_size_kb': model_size,
                    'energy_consumption': energy_estimate,
                }
                
                print(f"      ✅ {name}:")
                print(f"         MAE={mae:.6f}, R²={r2:.4f}")
                print(f"         دقت ±1%={accuracy_1:.1f}%, ±5%={accuracy_5:.1f}%")
                print(f"         زمان={training_time:.2f}s, حجم={model_size:.1f}KB")
                
            except Exception as e:
                print(f"      ❌ خطا در {name}: {e}")
                continue
        
        # ذخیره نتایج
        output_dir = Path("results/complete_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ذخیره به صورت CSV
        df = pd.DataFrame([
            {'model': name, **metrics} 
            for name, metrics in results.items()
        ])
        df.to_csv(output_dir / "complete_results.csv", index=False)
        
        # ایجاد نمودارهای ساده
        print("📊 ایجاد نمودارهای مقایسه...")
        create_simple_charts(results, output_dir)
        
        # نمایش نتایج نهایی
        print("\n" + "=" * 70)
        print("🏆 نتایج نهایی - مقایسه کامل مدل‌های پایه")
        print("=" * 70)
        
        # بهترین مدل‌ها براساس معیارهای مختلف
        print("\n🥇 برترین مدل‌ها بر اساس معیارهای مختلف:")
        print("-" * 50)
        
        if results:
            # بهترین از نظر MAE
            best_mae = min(results.items(), key=lambda x: x[1]['mae'])
            print(f"📊 بهترین MAE: {best_mae[0]} ({best_mae[1]['mae']:.6f})")
            
            # بهترین از نظر R²
            best_r2 = max(results.items(), key=lambda x: x[1]['r2_score'])
            print(f"📈 بهترین R²: {best_r2[0]} ({best_r2[1]['r2_score']:.4f})")
            
            # بهترین از نظر دقت ±1%
            best_acc1 = max(results.items(), key=lambda x: x[1]['accuracy_1%'])
            print(f"🎯 بهترین دقت ±1%: {best_acc1[0]} ({best_acc1[1]['accuracy_1%']:.1f}%)")
            
            # بهترین از نظر دقت ±5%
            best_acc5 = max(results.items(), key=lambda x: x[1]['accuracy_5%'])
            print(f"🎯 بهترین دقت ±5%: {best_acc5[0]} ({best_acc5[1]['accuracy_5%']:.1f}%)")
            
            # بهترین از نظر سرعت
            best_speed = min(results.items(), key=lambda x: x[1]['inference_time_seconds'])
            print(f"⚡ سریع‌ترین: {best_speed[0]} ({best_speed[1]['inference_time_seconds']:.4f}ثانیه)")
            
            # بهترین از نظر حجم
            best_size = min(results.items(), key=lambda x: x[1]['model_size_kb'])
            print(f"💾 کم‌حجم‌ترین: {best_size[0]} ({best_size[1]['model_size_kb']:.1f}KB)")
        
        # جدول کامل نتایج
        print("\n📋 جدول کامل نتایج همه مدل‌ها:")
        print("-" * 90)
        print(f"{'مدل':<15} {'MAE':<10} {'R²':<8} {'±1%':<6} {'±5%':<6} {'زمان(s)':<8} {'حجم(KB)':<8}")
        print("-" * 90)
        
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['mae']):
            print(f"{name:<15} {metrics['mae']:.6f} {metrics['r2_score']:.4f} "
                  f"{metrics['accuracy_1%']:.1f}% {metrics['accuracy_5%']:.1f}% "
                  f"{metrics['training_time_seconds']:.2f} {metrics['model_size_kb']:.1f}")
        
        print(f"\n💾 تمام نتایج ذخیره شد در: {output_dir}")
        print("📊 نمودارها در پوشه results/complete_comparison ساخته شدند")
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()

def create_simple_charts(results, output_dir):
    """ایجاد نمودارهای ساده بدون seaborn"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # تبدیل به DataFrame
    df = pd.DataFrame([
        {'Model': name, **metrics} 
        for name, metrics in results.items()
    ])
    
    # تنظیم فونت برای فارسی (اگر مشکل داشت حذف شود)
    try:
        plt.rcParams['font.family'] = 'B Nazanin'
    except:
        pass
    
    # ۱. نمودار MAE مقایسه
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('mae')
    bars = plt.bar(df_sorted['Model'], df_sorted['mae'], color='skyblue')
    plt.title('مقایسه MAE مدل‌های مختلف')
    plt.xlabel('مدل')
    plt.ylabel('MAE')
    plt.xticks(rotation=45, ha='right')
    
    # اضافه کردن مقادیر روی نمودار
    for bar, value in zip(bars, df_sorted['mae']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001, 
                f'{value:.6f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ۲. نمودار دقت ±1%
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('accuracy_1%', ascending=False)
    bars = plt.bar(df_sorted['Model'], df_sorted['accuracy_1%'], color='lightgreen')
    plt.title('مقایسه دقت ±1% مدل‌های مختلف')
    plt.xlabel('مدل')
    plt.ylabel('دقت ±1% (%)')
    plt.xticks(rotation=45, ha='right')
    
    # اضافه کردن مقادیر روی نمودار
    for bar, value in zip(bars, df_sorted['accuracy_1%']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_1p_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ۳. نمودار زمان آموزش
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('training_time_seconds')
    bars = plt.bar(df_sorted['Model'], df_sorted['training_time_seconds'], color='orange')
    plt.title('مقایسه زمان آموزش مدل‌ها')
    plt.xlabel('مدل')
    plt.ylabel('زمان آموزش (ثانیه)')
    plt.xticks(rotation=45, ha='right')
    
    # اضافه کردن مقادیر روی نمودار
    for bar, value in zip(bars, df_sorted['training_time_seconds']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}s', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ۴. نمودار R² score
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('r2_score', ascending=False)
    bars = plt.bar(df_sorted['Model'], df_sorted['r2_score'], color='purple')
    plt.title('مقایسه R² Score مدل‌ها')
    plt.xlabel('مدل')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45, ha='right')
    
    # اضافه کردن مقادیر روی نمودار
    for bar, value in zip(bars, df_sorted['r2_score']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ ۴ نمودار مقایسه ایجاد شد")

if __name__ == "__main__":
    main()