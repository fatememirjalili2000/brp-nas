# run_final_baselines.py
import sys
import os
sys.path.append('.')

def main():
    """برنامه اصلی برای اجرای کامل مدل‌های پایه"""
    try:
        from eagle.predictors.embedding_baselines.embedding_predictor import EmbeddingPredictor
        from eagle.models import nasbench201
        import pickle
        
        print("🚀 شروع ارزیابی جامع مدل‌های پایه")
        print("=" * 50)
        
        # بارگذاری دیتاست
        dataset_path = "results/desktop-cpu-core-i7-7820x.pickle"
        print(f"📁 در حال بارگذاری دیتاست از: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # تبدیل به لیست
        data_pairs = [(graph, latency) for graph, latency in dataset.items()]
        
        print(f"📊 تعداد کل داده‌ها: {len(data_pairs)}")
        
        # تقسیم داده‌ها
        train_size = 800  # داده آموزشی
        test_size = 200   # داده تست
        
        train_data = data_pairs[:train_size]
        test_data = data_pairs[train_size:train_size + test_size]
        
        print(f"🔢 داده آموزشی: {len(train_data)}")
        print(f"🔢 داده تست: {len(test_data)}")
        
        # ایجاد پیش‌بین
        predictor_args = {'embedding_dim': 600}
        predictor = EmbeddingPredictor(predictor_args)
        
        print("🎯 شروع آموزش مدل‌ها...")
        predictor.train_models(train_data, nasbench201)
        
        print("📊 شروع ارزیابی مدل‌ها...")
        results = predictor.evaluate_models(test_data, nasbench201)
        
        print("✅ ارزیابی با موفقیت انجام شد!")
        
        # نمایش نتایج
        print("\n" + "=" * 50)
        print("🏆 نتایج نهایی:")
        print("=" * 50)
        
        for name, metrics in results.items():
            print(f"📈 {name}:")
            print(f"   📍 MAE: {metrics['mae']:.6f}")
            print(f"   📍 R²: {metrics['r2_score']:.4f}")
            print(f"   📍 دقت ±۱٪: {metrics['accuracy_1%']:.1f}٪")
            print(f"   ⏱️ زمان پیش‌بینی: {metrics['inference_time_seconds']:.4f} ثانیه")
            print()
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()