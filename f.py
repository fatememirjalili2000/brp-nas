import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# تنظیمات اولیه
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

def create_simple_comparison():
    """ایجاد نمودارهای ساده برای مقایسه نتایج"""
    
    # داده‌های نتایج از گزارش شما
    data = {
        'model': [
            'GCN_Original', 'ExtraTrees', 'RandomForest', 'GradientBoosting', 
            'DecisionTree', 'CatBoost', 'KNeighbors', 'AdaBoost',
            'Ensemble1_TopMAE', 'Ensemble2_Diverse', 'Ensemble3_AllBest', 'Ensemble4_Stacking'
        ],
        'mae': [
            0.000573, 0.000118, 0.000139, 0.000135,
            0.000140, 0.000169, 0.000174, 0.000182,
            0.000125, 0.000141, 0.000130, 0.000125
        ],
        'accuracy_5%': [
            27.8, 82.7, 80.5, 78.4,
            83.2, 76.6, 70.5, 70.0,
            81.8, 75.3, 76.5, 80.2
        ],
        'r2_score': [
            0.7500, 0.9818, 0.9771, 0.9789,
            0.9728, 0.9700, 0.9637, 0.9744,
            0.9802, 0.9781, 0.9802, 0.9812
        ],
        'type': [
            'original', 'base', 'base', 'base', 'base', 'base', 'base', 'base',
            'ensemble', 'ensemble', 'ensemble', 'ensemble'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # ایجاد دایرکتوری خروجی
    output_dir = Path("simple_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # 1. نمودار MAE
    plt.figure(figsize=(14, 8))
    colors = ['red' if t == 'original' else 'blue' if t == 'base' else 'green' for t in df['type']]
    
    bars = plt.bar(df['model'], df['mae'], color=colors, alpha=0.7, edgecolor='black')
    plt.title('مقایسه MAE: مدل اصلی، مدل‌های پایه و انسمل‌ها', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('مدل‌ها', fontsize=12)
    plt.ylabel('میانگین خطای مطلق (MAE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')  # استفاده از مقیاس لگاریتمی برای نمایش بهتر
    
    # اضافه کردن مقادیر روی نمودار
    for bar, value in zip(bars, df['mae']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{value:.6f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # اضافه کردن راهنما
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='مدل اصلی GCN'),
        Patch(facecolor='blue', alpha=0.7, label='مدل‌های پایه'),
        Patch(facecolor='green', alpha=0.7, label='انسمل‌ها')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. نمودار دقت ±5%
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['model'], df['accuracy_5%'], color=colors, alpha=0.7, edgecolor='black')
    plt.title('مقایسه دقت ±5%: همه مدل‌ها', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('مدل‌ها', fontsize=12)
    plt.ylabel('دقت ±5% (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # اضافه کردن مقادیر روی نمودار
    for bar, value in zip(bars, df['accuracy_5%']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # خط برای دقت GCN اصلی
    plt.axhline(y=27.8, color='red', linestyle='--', alpha=0.8, label='دقت GCN اصلی')
    
    plt.legend(handles=legend_elements + [plt.Line2D([0], [0], color='red', linestyle='--', label='دقت GCN اصلی')])
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. نمودار R² score
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['model'], df['r2_score'], color=colors, alpha=0.7, edgecolor='black')
    plt.title('مقایسه R² Score: همه مدل‌ها', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('مدل‌ها', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.7, 1.0)
    
    # اضافه کردن مقادیر روی نمودار
    for bar, value in zip(bars, df['r2_score']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. نمودار بهبود نسبت به GCN اصلی
    gcn_mae = df[df['model'] == 'GCN_Original']['mae'].values[0]
    improvements = [(gcn_mae - mae) / gcn_mae * 100 for mae in df['mae']]
    
    plt.figure(figsize=(14, 8))
    colors_improve = ['gray' if t == 'original' else 'blue' if t == 'base' else 'green' for t in df['type']]
    
    bars = plt.bar(df['model'], improvements, color=colors_improve, alpha=0.7, edgecolor='black')
    plt.title('درصد بهبود MAE نسبت به مدل GCN اصلی', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('مدل‌ها', fontsize=12)
    plt.ylabel('درصد بهبود (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # اضافه کردن مقادیر روی نمودار
    for bar, value in zip(bars, improvements):
        if value > 0:  # فقط برای مدل‌های بهتر از GCN
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. نمودار رادار برای ۵ مدل برتر
    top_models = df[df['model'] != 'GCN_Original'].nlargest(5, 'accuracy_5%')
    
    # نرمال‌سازی داده‌ها برای نمودار رادار
    categories = ['MAE (معکوس)', 'دقت ±5%', 'R² Score']
    
    def normalize_data(values, reverse=False):
        min_val = min(values)
        max_val = max(values)
        if reverse:
            return [(max_val - v) / (max_val - min_val) for v in values]
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    # محاسبه مقادیر نرمال‌شده
    mae_norm = normalize_data(top_models['mae'], reverse=True)
    acc_norm = normalize_data(top_models['accuracy_5%'])
    r2_norm = normalize_data(top_models['r2_score'])
    
    # ایجاد نمودار رادار
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # بستن دایره
    
    colors_radar = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (_, row) in enumerate(top_models.iterrows()):
        values = [mae_norm[i], acc_norm[i], r2_norm[i]]
        values += values[:1]  # بستن دایره
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors_radar[i])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    plt.title('۵ مدل برتر - نمودار رادار', size=16, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(output_dir / 'top5_radar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ تمام نمودارها با موفقیت ایجاد شدند!")
    print(f"📁 نمودارها در پوشه {output_dir} ذخیره شدند")
    
    # نمایش خلاصه نتایج
    print("\n🏆 خلاصه نتایج:")
    print("=" * 50)
    print(f"بهترین مدل پایه: {df[df['type'] == 'base'].loc[df[df['type'] == 'base']['mae'].idxmin()]['model']}")
    print(f"بهترین انسمل: {df[df['type'] == 'ensemble'].loc[df[df['type'] == 'ensemble']['mae'].idxmin()]['model']}")
    print(f"بیشترین بهبود: {max(improvements):.1f}%")
    print(f"بهترین دقت ±5%: {df['accuracy_5%'].max():.1f}%")

if __name__ == "__main__":
    create_simple_comparison()