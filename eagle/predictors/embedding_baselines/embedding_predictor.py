# eagle/predictors/embedding_baselines/embedding_predictor.py
import torch
import torch.nn as nn
import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, kendalltau
import pandas as pd

class EmbeddingPredictor(nn.Module):
    """
    Predictor that uses GCN embeddings with traditional ML models
    """
    
    def __init__(self, predictor_args=None):
        super().__init__()
        self.predictor_args = predictor_args or {}  # تضمین می‌کند هیچگاه None نباشد
        self.binary_classifier = False
        self.models = {}
        self.current_model = None
        self.embedding_dim = self.predictor_args.get('embedding_dim', 600)  # حالا safe است
        
        print(f"✅ EmbeddingPredictor initialized with dim: {self.embedding_dim}")
        
    def extract_embeddings_single(self, model_module, graph):
        """Extract embedding for a single graph"""
        try:
            from .. import infer
            
            # Create dummy target for compatibility
            dummy_target = 0.0
            
            # Prepare tensors
            adjacency, features, _, _ = infer.prepare_tensors(
                [graph], [dummy_target], model_module, False, False
            )
            
            # Load or create GCN model for feature extraction
            gcn_model = self.get_gcn_model()
            gcn_model.eval()
            
            with torch.no_grad():
                if hasattr(gcn_model, 'extract_features'):
                    emb = gcn_model.extract_features(adjacency, features)
                else:
                    # Fallback method
                    x = gcn_model.forward_single_model(adjacency, features)
                    emb = x[:, 0]  # Global node
                
                return emb.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"⚠️ Error extracting embedding: {e}")
            # Fallback: return random embedding
            return np.random.randn(self.embedding_dim)
    
    def get_gcn_model(self):
        """Get GCN model for feature extraction"""
        try:
            from ..gcn.gcn import GCN
            # Create a GCN model with same architecture as trained one
            model = GCN(
                num_features=5, 
                num_hidden=self.embedding_dim, 
                num_layers=4
            )
            
            # Try to load trained weights
            model_path = Path("results/nasbench201/latency/cpu/gcn/predictor.pt")
            if model_path.exists():
                model.load_state_dict(torch.load(model_path))
                print("✅ Loaded pre-trained GCN weights")
            else:
                print("⚠️ No pre-trained GCN found, using untrained model")
            
            if torch.cuda.is_available():
                model.cuda()
                
            return model
            
        except Exception as e:
            print(f"⚠️ Could not load GCN model: {e}")
            # Return a simple model as fallback
            from ..gcn.gcn import GCN
            model = GCN(num_features=5, num_hidden=self.embedding_dim, num_layers=4)
            if torch.cuda.is_available():
                model.cuda()
            return model
    
    def prepare_baseline_models(self):
        """Initialize baseline models"""
        self.baseline_models = {
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10),
            'XGBoost': XGBRegressor(n_estimators=50, random_state=42, max_depth=6),
            'LinearRegression': LinearRegression(),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'SVR': SVR(kernel='rbf', C=1.0),
        }
        print(f"✅ Prepared {len(self.baseline_models)} baseline models")
    
    def train_models(self, train_dataset, model_module):
        """Train all baseline models"""
        print("🔍 Extracting embeddings for training...")
        
        # Extract embeddings
        X_train = []
        y_train = []
        
        for i, (graph, target) in enumerate(train_dataset):
            if i % 100 == 0:  # گزارش پیشرفت بیشتر
                print(f"📊 Processed {i}/{len(train_dataset)} training samples...")
            
            embedding = self.extract_embeddings_single(model_module, graph)
            X_train.append(embedding)
            y_train.append(target)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"✅ Training embeddings shape: {X_train.shape}")
        
        # Prepare and train models
        self.prepare_baseline_models()
        self.trained_models = {}
        
        for name, model in self.baseline_models.items():
            print(f"🎯 Training {name}...")
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                self.trained_models[name] = model
                print(f"   ✅ {name} trained in {training_time:.2f}s")
            except Exception as e:
                print(f"   ❌ Failed to train {name}: {e}")
        
        # Store training data for later use
        self.X_train = X_train
        self.y_train = y_train
        
        print(f"✅ Successfully trained {len(self.trained_models)} models")
        return self.trained_models
    
    def evaluate_models(self, test_dataset, model_module):
        """Evaluate all trained models"""
        if not hasattr(self, 'trained_models') or not self.trained_models:
            raise ValueError("No models trained! Call train_models first.")
        
        print("🔍 Extracting embeddings for testing...")
        
        # Extract test embeddings
        X_test = []
        y_test = []
        
        for i, (graph, target) in enumerate(test_dataset):
            if i % 50 == 0:  # گزارش پیشرفت بیشتر
                print(f"📊 Processed {i}/{len(test_dataset)} test samples...")
            
            embedding = self.extract_embeddings_single(model_module, graph)
            X_test.append(embedding)
            y_test.append(target)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"✅ Test embeddings shape: {X_test.shape}")
        
        # Evaluate each model
        results = {}
        for name, model in self.trained_models.items():
            print(f"📈 Evaluating {name}...")
            
            try:
                # Inference
                start_time = time.time()
                y_pred = model.predict(X_test)
                inference_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred, inference_time)
                results[name] = metrics
                
                print(f"   ✅ {name}: MAE={metrics['mae']:.4f}, R²={metrics['r2_score']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed to evaluate {name}: {e}")
                continue
        
        # Save results
        self.save_results(results)
        return results
    
    def calculate_metrics(self, y_true, y_pred, inference_time):
        """Calculate comprehensive metrics"""
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Accuracy within tolerance (like original project)
        tolerances = [0.01, 0.05, 0.10, 0.20]
        accuracy_metrics = {}
        for tol in tolerances:
            within_tolerance = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) <= tol)
            accuracy_metrics[f'accuracy_{int(tol*100)}%'] = within_tolerance * 100
        
        # Ranking metrics
        try:
            spearman_corr, _ = spearmanr(y_true, y_pred)
            kendall_tau, _ = kendalltau(y_true, y_pred)
        except:
            spearman_corr = 0
            kendall_tau = 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'spearman_corr': spearman_corr,
            'kendall_tau': kendall_tau,
            'inference_time_seconds': inference_time,
            **accuracy_metrics
        }
    
    def save_results(self, results):
        """Save evaluation results"""
        output_dir = Path("results/nasbench201/latency/cpu/embedding_baselines")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df = pd.DataFrame([
            {'model': name, **metrics} 
            for name, metrics in results.items()
        ])
        df.to_csv(output_dir / "baseline_results.csv", index=False)
        
        # Save detailed results
        with open(output_dir / "detailed_results.pickle", 'wb') as f:
            pickle.dump(results, f)
        
        # Generate report
        self.generate_report(df, output_dir)
        
        print(f"💾 Results saved to: {output_dir}")
    
    def generate_report(self, df, output_dir):
        """Generate summary report"""
        report = f"""
# Embedding Baseline Models Report

## Best Models:

### By MAE (Lower is Better):
{df.nsmallest(3, 'mae')[['model', 'mae', 'r2_score']].to_string(index=False)}

### By R² Score (Higher is Better):
{df.nlargest(3, 'r2_score')[['model', 'r2_score', 'mae']].to_string(index=False)}

### By Inference Speed:
{df.nsmallest(3, 'inference_time_seconds')[['model', 'inference_time_seconds', 'mae']].to_string(index=False)}

## All Results:
{df.to_string(index=False)}

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_dir / "summary_report.md", 'w') as f:
            f.write(report)
    
    def forward(self, adjacency, features, augments=None):
        """Compatibility method - uses the first trained model for prediction"""
        if not hasattr(self, 'trained_models') or not self.trained_models:
            raise ValueError("No models trained! Call train_models first.")
        
        # Use the first available model
        model_name = list(self.trained_models.keys())[0]
        model = self.trained_models[model_name]
        self.current_model = model_name
        
        # Convert tensor to numpy for sklearn prediction
        if hasattr(features, 'cpu'):
            features = features.cpu().numpy()
        
        # Assuming features already contain the embeddings
        if features.ndim > 2:
            # Flatten if needed
            features = features.reshape(features.shape[0], -1)
        
        predictions = model.predict(features)
        return torch.tensor(predictions, dtype=features.dtype).unsqueeze(1)
    
    def reset_last(self):
        """Compatibility method"""
        pass
    
    def final_params(self):
        """Compatibility method"""
        return []


def get_predictor(predictor_args=None):
    """Factory function for creating embedding predictor"""
    return EmbeddingPredictor(predictor_args)