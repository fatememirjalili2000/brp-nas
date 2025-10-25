

import os
import pickle
import pathlib
import argparse
import importlib
import functools
import contextlib
import statistics
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from . import utils
from . import infer
from . import dataset as dataset_mod

# Import system monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. CPU and memory metrics will not be collected.")
try:
    import pyRAPL
    PYRAPL_AVAILABLE = True
except ImportError:
    PYRAPL_AVAILABLE = False
    print("Warning: pyRAPL not available. Energy consumption metrics will not be collected.")


def set_seed(seed=42):
    """Set seed for reproducibility across different systems"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (CPU or GPU) automatically"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class ModelComplexityAnalyzer:
    """Class to analyze model complexity including FLOPS and parameters"""
    def __init__(self):
        self.model_flops = {}
        self.model_parameters = {}
    
    def compute_model_complexity(self, model, sample_input):
        """Compute FLOPS and parameters for a model"""
        try:
            # Compute number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            self.model_parameters['total'] = total_params
            self.model_parameters['trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate FLOPS (this is a simplified estimation)
            flops_estimate = self.estimate_flops(model, sample_input)
            self.model_flops['estimated'] = flops_estimate
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': self.model_parameters['trainable'],
                'estimated_flops': flops_estimate
            }
        except Exception as e:
            print(f"Warning: Could not compute model complexity: {e}")
            return {
                'total_parameters': 0,
                'trainable_parameters': 0,
                'estimated_flops': 0
            }
    
    def estimate_flops(self, model, sample_input):
        """Estimate FLOPS for the model (simplified version)"""
        try:
            if hasattr(model, 'estimate_flops'):
                return model.estimate_flops(sample_input)
            
            total_flops = 0
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    if hasattr(module, 'weight'):
                        in_features = module.in_features
                        out_features = module.out_features
                        total_flops += 2 * in_features * out_features
                elif isinstance(module, nn.Conv2d):
                    if hasattr(module, 'weight'):
                        kernel_size = module.kernel_size[0] * module.kernel_size[1]
                        in_channels = module.in_channels
                        out_channels = module.out_channels
                        output_size = 112
                        total_flops += 2 * kernel_size * in_channels * out_channels * output_size * output_size
            
            return total_flops
        except:
            return 0


class MetricsCollector:
    """Class to collect and compute performance and system metrics during training"""
    
    def __init__(self, experiment_id=None):
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.experiment_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_mae = []
        self.val_mae = []
        self.train_r2 = []
        self.val_r2 = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epoch_times = []
        
        # System metrics
        self.memory_usage = []
        self.cpu_usage = []
        self.energy_consumption = []
        
        # Model complexity metrics
        self.model_complexity = {}
        self.inference_times = []
        
        # For final results
        self.best_val_targets = []
        self.best_val_predictions = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        
        # Test set metrics
        self.test_targets = []
        self.test_predictions = []
        self.test_mae = 0.0
        self.test_r2 = 0.0
        self.test_accuracies = [0.0, 0.0, 0.0, 0.0]
        
        # Energy measurement setup
        if PYRAPL_AVAILABLE:
            pyRAPL.setup()
            self.energy_meter = pyRAPL.Measurement('training')
        
        # Initialize complexity analyzer
        self.complexity_analyzer = ModelComplexityAnalyzer()
    
    def set_model_complexity(self, complexity_info):
        """Set model complexity information"""
        self.model_complexity = complexity_info
    
    def start_epoch(self):
        """Start measurement for a new epoch"""
        self.epoch_start_time = time.time()
        if PYRAPL_AVAILABLE:
            self.energy_meter.begin()
        
        if PSUTIL_AVAILABLE:
            self.initial_memory = psutil.virtual_memory().used
            self.initial_cpu = psutil.cpu_percent(interval=None)
    
    def end_epoch(self):
        """End measurement for current epoch"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        if PSUTIL_AVAILABLE:
            current_memory = psutil.virtual_memory().used
            memory_used_mb = (current_memory - self.initial_memory) / (1024 * 1024)
            self.memory_usage.append(max(0, memory_used_mb))
            
            current_cpu = psutil.cpu_percent(interval=None)
            self.cpu_usage.append((self.initial_cpu + current_cpu) / 2)
        
        if PYRAPL_AVAILABLE:
            self.energy_meter.end()
            energy_joules = sum(self.energy_meter.result.energy)
            self.energy_consumption.append(energy_joules)
    
    def record_inference_time(self, inference_time):
        """Record inference time for a batch"""
        self.inference_times.append(inference_time)
    
    def compute_regression_metrics(self, targets, predictions, leeways=[0.01, 0.05, 0.1, 0.2]):
        """Compute regression metrics with proper R² calculation"""
        if not targets or not predictions:
            return 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], []
            
        targets_np = np.array(targets, dtype=np.float64)
        predictions_np = np.array(predictions, dtype=np.float64)
        
        # Handle very small values and zero values
        if np.max(np.abs(targets_np)) < 1e-10:
            targets_np = targets_np + 1e-10
            predictions_np = predictions_np + 1e-10
        
        # MAE calculation
        mae = mean_absolute_error(targets_np, predictions_np)
        
        # Fixed R² calculation with sklearn
        if len(targets_np) < 2:
            r2 = 0.0
        else:
            try:
                r2 = r2_score(targets_np, predictions_np)
                # Clamp R² to reasonable range
                r2 = max(min(r2, 1.0), -1.0)
            except Exception as e:
                print(f"Warning in R² calculation: {e}")
                r2 = 0.0
        
        # Accuracy within error bounds with numerical stability
        with np.errstate(divide='ignore', invalid='ignore'):
            # Avoid division by zero
            safe_targets = np.where(np.abs(targets_np) < 1e-10, 1e-10, targets_np)
            relative_errors = np.abs((predictions_np - targets_np) / np.abs(safe_targets))
            relative_errors = np.nan_to_num(relative_errors, nan=1.0, posinf=1.0, neginf=1.0)
        
        accuracies = []
        for leeway in leeways:
            accuracy = np.mean(relative_errors <= leeway)
            accuracies.append(float(accuracy))
        
        return float(mae), float(r2), accuracies, relative_errors.tolist()
    
    def record_train_metrics(self, loss, targets, predictions):
        """Record training metrics for current epoch"""
        self.train_losses.append(float(loss))
        if targets and predictions:
            mae, r2, accuracies, _ = self.compute_regression_metrics(targets, predictions)
            self.train_mae.append(mae)
            self.train_r2.append(r2)
            self.train_accuracies.append(accuracies)
    
    def record_val_metrics(self, loss, targets, predictions, epoch):
        """Record validation metrics for current epoch and track best epoch"""
        self.val_losses.append(float(loss))
        
        if targets and predictions:
            mae, r2, accuracies, _ = self.compute_regression_metrics(targets, predictions)
            self.val_mae.append(mae)
            self.val_r2.append(r2)
            self.val_accuracies.append(accuracies)
            
            if loss < self.best_val_loss:
                self.best_val_loss = loss
                self.best_epoch = epoch
                self.best_val_targets = targets.copy()
                self.best_val_predictions = predictions.copy()
    
    def record_test_metrics(self, targets, predictions):
        """Record test set metrics"""
        self.test_targets = targets.copy()
        self.test_predictions = predictions.copy()
        if targets and predictions:
            self.test_mae, self.test_r2, self.test_accuracies, _ = self.compute_regression_metrics(targets, predictions)
    
    def get_final_results(self, cfg, model_name, total_training_time):
        """Compile final results dictionary"""
        if self.best_val_targets and self.best_val_predictions:
            final_mae, final_r2, final_accuracies, abs_errors = self.compute_regression_metrics(
                self.best_val_targets, self.best_val_predictions
            )
        else:
            final_mae = self.val_mae[-1] if self.val_mae else 0.0
            final_r2 = self.val_r2[-1] if self.val_r2 else 0.0
            final_accuracies = self.val_accuracies[-1] if self.val_accuracies else [0.0, 0.0, 0.0, 0.0]
            abs_errors = []
        
        total_inference_time = sum(self.inference_times) if self.inference_times else sum(self.epoch_times)
        num_samples = len(self.best_val_targets) if self.best_val_targets else 1
        avg_inference_time = total_inference_time / max(1, num_samples)
        
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0.0
        avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0.0
        total_energy = sum(self.energy_consumption) if self.energy_consumption else 0.0
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0.0
        
        results = {
            'experiment_id': self.experiment_id,
            'timestamp': self.experiment_timestamp,
            'model_name': model_name,            
            'model_complexity': self.model_complexity,
            
            # Validation metrics
            'mae': final_mae,
            'r2': final_r2,
            'accuracy_1%': final_accuracies[0] if final_accuracies else 0.0,
            'accuracy_5%': final_accuracies[1] if final_accuracies else 0.0,
            'accuracy_10%': final_accuracies[2] if final_accuracies else 0.0,
            'accuracy_20%': final_accuracies[3] if final_accuracies else 0.0,
            
            # Test metrics
            'test_mae': self.test_mae,
            'test_r2': self.test_r2,
            'test_accuracy_1%': self.test_accuracies[0] if self.test_accuracies else 0.0,
            'test_accuracy_5%': self.test_accuracies[1] if self.test_accuracies else 0.0,
            'test_accuracy_10%': self.test_accuracies[2] if self.test_accuracies else 0.0,
            'test_accuracy_20%': self.test_accuracies[3] if self.test_accuracies else 0.0,
            
            # Time metrics
            'total_training_time_seconds': total_training_time,
            'average_epoch_time_seconds': avg_epoch_time,
            'average_inference_time_seconds': avg_inference_time,
            'total_inference_time_seconds': total_inference_time,
            
            # System metrics
            'average_memory_usage_mb': avg_memory,
            'average_cpu_percent': avg_cpu,
            'total_energy_joules': total_energy,
            'average_energy_per_epoch_joules': total_energy / max(1, len(self.energy_consumption)),
            
            # Training configuration
            'epochs': cfg.get('epochs', 0),
            'learning_rate': cfg.get('learning_rate', 0.0),
            'batch_size': cfg.get('batch_size', 0),
            'weight_decay': cfg.get('weight_decay', 0.0),            
            
            # Training progress
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else float('nan'),
            'final_val_loss': self.val_losses[-1] if self.val_losses else float('nan'),
            'best_val_loss': self.best_val_loss,
            
            # Complete history
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mae': self.train_mae,
            'val_mae': self.val_mae,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'epoch_times': self.epoch_times,
            'inference_times': self.inference_times,
            'memory_usage_history': self.memory_usage,
            'cpu_usage_history': self.cpu_usage,
            'energy_consumption_history': self.energy_consumption
        }
        
        return results
    
    def save_comparison_data(self, results, outdir, model_name, exp_name=None):
        """Save data in a format suitable for future comparisons"""
        comparison_dir = outdir / "comparison_data"
        comparison_dir.mkdir(exist_ok=True)
        
        filename = f"comparison_{model_name}_{self.experiment_id}"
        if exp_name:
            filename += f"_{exp_name}"
        filename += ".json"
        
        filepath = comparison_dir / filename
        
        def convert_types(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_types(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.save_comparison_summary(results, comparison_dir)
        
        return filepath
    
    def save_comparison_summary(self, results, comparison_dir):
        """Save a summary CSV file for easy comparison across experiments"""
        import csv
        
        summary_file = comparison_dir / "experiment_summary.csv"
        
        write_header = not summary_file.exists()
        
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow([
                    'experiment_id', 'timestamp', 'model_name', 
                    'val_mae', 'val_r2', 'val_accuracy_1%', 'val_accuracy_5%', 'val_accuracy_10%', 'val_accuracy_20%',
                    'test_mae', 'test_r2', 'test_accuracy_1%', 'test_accuracy_5%', 'test_accuracy_10%', 'test_accuracy_20%',
                    'total_training_time_seconds', 'average_inference_time_seconds',
                    'total_parameters', 'trainable_parameters', 'estimated_flops',
                    'average_memory_usage_mb', 'average_cpu_percent', 'total_energy_joules', 'best_epoch',
                    'epochs', 'learning_rate', 'batch_size'
                ])
            
            complexity = results.get('model_complexity', {})
            total_params = complexity.get('total_parameters', 0)
            trainable_params = complexity.get('trainable_parameters', 0)
            flops = complexity.get('estimated_flops', 0)
            
            writer.writerow([
                results['experiment_id'],
                results['timestamp'],
                results['model_name'],
                results['mae'],
                results['r2'],
                results['accuracy_1%'],
                results['accuracy_5%'],
                results['accuracy_10%'],
                results['accuracy_20%'],
                results['test_mae'],
                results['test_r2'],
                results['test_accuracy_1%'],
                results['test_accuracy_5%'],
                results['test_accuracy_10%'],
                results['test_accuracy_20%'],
                results['total_training_time_seconds'],
                results['average_inference_time_seconds'],
                total_params,
                trainable_params,
                flops,
                results['average_memory_usage_mb'],
                results['average_cpu_percent'],
                results['total_energy_joules'],
                results['best_epoch'],
                results['epochs'],
                results['learning_rate'],
                results['batch_size']
            ])
    
    def save_results(self, results, outdir, model_name, exp_name=None):
        """Save results to JSON file"""
        filename = f"results_{model_name}_{self.experiment_id}"
        if exp_name:
            filename += f"_{exp_name}"
        filename += ".json"
        
        filepath = outdir / filename
        
        def convert_types(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_types(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.save_comparison_data(results, outdir, model_name, exp_name)
        
        return filepath
    
    def create_charts(self, results, outdir, model_name, exp_name=None):
        """Create comprehensive charts from the collected metrics"""
        try:
            charts_dir = outdir / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            # 1. Training and Validation Loss
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            epochs_range = range(1, len(self.train_losses) + 1)
            plt.plot(epochs_range, self.train_losses, label='Training Loss')
            if self.val_losses:
                plt.plot(epochs_range, self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # 2. MAE and R²
            plt.subplot(1, 2, 2)
            if self.train_mae and self.val_mae:
                epochs_range_metrics = range(1, min(len(self.train_mae), len(self.val_mae)) + 1)
                plt.plot(epochs_range_metrics, self.train_mae[:len(epochs_range_metrics)], label='Train MAE')
                plt.plot(epochs_range_metrics, self.val_mae[:len(epochs_range_metrics)], label='Val MAE')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.title('Mean Absolute Error')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            loss_chart_path = charts_dir / f"training_metrics_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
            plt.savefig(loss_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Accuracy Metrics
            if self.val_accuracies:
                plt.figure(figsize=(10, 6))
                accuracies = np.array(self.val_accuracies)
                epochs_range_acc = range(1, len(accuracies) + 1)
                
                plt.plot(epochs_range_acc, accuracies[:, 0], label='±1% Accuracy', marker='o')
                plt.plot(epochs_range_acc, accuracies[:, 1], label='±5% Accuracy', marker='s')
                plt.plot(epochs_range_acc, accuracies[:, 2], label='±10% Accuracy', marker='^')
                plt.plot(epochs_range_acc, accuracies[:, 3], label='±20% Accuracy', marker='d')
                
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Validation Accuracy at Different Error Bounds')
                plt.legend()
                plt.grid(True)
                
                accuracy_chart_path = charts_dir / f"accuracy_metrics_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
                plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. System Metrics
            if self.epoch_times and self.memory_usage and self.cpu_usage:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.plot(epochs_range, self.epoch_times)
                plt.xlabel('Epoch')
                plt.ylabel('Time (s)')
                plt.title('Epoch Duration')
                plt.grid(True)
                
                plt.subplot(2, 2, 2)
                plt.plot(epochs_range, self.memory_usage)
                plt.xlabel('Epoch')
                plt.ylabel('Memory (MB)')
                plt.title('Memory Usage')
                plt.grid(True)
                
                plt.subplot(2, 2, 3)
                plt.plot(epochs_range, self.cpu_usage)
                plt.xlabel('Epoch')
                plt.ylabel('CPU (%)')
                plt.title('CPU Usage')
                plt.grid(True)
                
                plt.subplot(2, 2, 4)
                if self.energy_consumption:
                    plt.plot(epochs_range, self.energy_consumption)
                    plt.xlabel('Epoch')
                    plt.ylabel('Energy (J)')
                    plt.title('Energy Consumption')
                else:
                    plt.text(0.5, 0.5, 'Energy data\nnot available', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                plt.grid(True)
                
                plt.tight_layout()
                system_chart_path = charts_dir / f"system_metrics_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
                plt.savefig(system_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 5. Final Prediction vs Target Scatter Plot (using BEST epoch data)
            if self.best_val_targets and self.best_val_predictions:
                plt.figure(figsize=(8, 6))
                plt.scatter(self.best_val_targets, self.best_val_predictions, alpha=0.6)
                
                min_val = min(min(self.best_val_targets), min(self.best_val_predictions))
                max_val = max(max(self.best_val_targets), max(self.best_val_predictions))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                plt.title(f'Best Epoch Predictions vs True Values\nMAE: {results["mae"]:.6f}, R²: {results["r2"]:.4f}')
                plt.grid(True)
                
                scatter_chart_path = charts_dir / f"predictions_scatter_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
                plt.savefig(scatter_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 6. Test set predictions scatter plot
            if self.test_targets and self.test_predictions:
                plt.figure(figsize=(8, 6))
                plt.scatter(self.test_targets, self.test_predictions, alpha=0.6)
                
                min_val = min(min(self.test_targets), min(self.test_predictions))
                max_val = max(max(self.test_targets), max(self.test_predictions))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                plt.title(f'Test Set Predictions vs True Values\nMAE: {results["test_mae"]:.6f}, R²: {results["test_r2"]:.4f}')
                plt.grid(True)
                
                test_scatter_path = charts_dir / f"test_predictions_scatter_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
                plt.savefig(test_scatter_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 7. Model Complexity Chart
            if self.model_complexity:
                plt.figure(figsize=(10, 6))
                metrics = ['Total Parameters', 'Trainable Parameters', 'Estimated FLOPS']
                values = [
                    self.model_complexity.get('total_parameters', 0),
                    self.model_complexity.get('trainable_parameters', 0),
                    self.model_complexity.get('estimated_flops', 0)
                ]
                
                values_normalized = [v / max(values) if max(values) > 0 else 0 for v in values]
                
                bars = plt.bar(metrics, values_normalized)
                plt.ylabel('Normalized Value')
                plt.title('Model Complexity Metrics')
                plt.xticks(rotation=45)
                
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:,}', ha='center', va='bottom')
                
                plt.tight_layout()
                complexity_chart_path = charts_dir / f"model_complexity_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
                plt.savefig(complexity_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Charts saved to: {charts_dir}")
            return charts_dir
            
        except Exception as e:
            print(f"Warning: Could not create charts: {e}")
            return None


def _train(model_module, model, gs, latencies, optimizer, criterion, device, normalize=False, augments=None):
    adjacency, features, latency, aug = infer.prepare_tensors(gs, latencies, model_module, model.binary_classifier, normalize, augments=augments)

    # Move tensors to device
    adjacency = adjacency.to(device)
    features = features.to(device)
    latency = latency.to(device)
    if aug is not None:
        aug = aug.to(device)

    model.train()
    optimizer.zero_grad()
    if augments is not None:
        predictions = model(adjacency, features, aug)
    else:
        predictions = model(adjacency, features)

    loss = criterion(predictions, latency)
    loss.backward()
    optimizer.step()

    return loss


def _test(model_module, model, g, latency, leeways, criterion, device, log_file=None, augments=None):
    if not model.binary_classifier:
        adjacency, features, latency, aug = infer.prepare_tensors([g], [latency], model_module, False, False, augments=augments)
    else:
        adjacency, features, latency, aug = infer.prepare_tensors(g, latency, model_module, model.binary_classifier, False, augments=augments)

    # Move tensors to device
    adjacency = adjacency.to(device)
    features = features.to(device)
    latency = latency.to(device)
    if aug is not None:
        aug = aug.to(device)

    torch.set_grad_enabled(False)
    model.eval()
    if augments is not None:
        predictions = model(adjacency, features, aug)
    else:
        predictions = model(adjacency, features)

    if not model.binary_classifier:
        if log_file is not None:
            log_file.write(f'{latency.item()} {predictions.item()} {g}\n')

    loss = criterion(predictions, latency)
    torch.set_grad_enabled(True)

    if not model.binary_classifier:
        results = []
        for l in leeways:
            results.append(utils.valid(predictions, latency, leeway=l))

        return results, loss, (latency.item(), predictions.item())
    else:
        return None, loss, None


def train(training_set,
        validation_set,
        outdir,
        device_name,
        model_name,
        metric,
        predictor_name,
        predictor,
        tensorboard,
        epochs,
        learning_rate,
        weight_decay,
        lr_patience,
        es_patience,
        batch_size,
        shuffle,
        optim_name,
        lr_scheduler,
        exp_name=None,
        reset_last=False,
        warmup=0,
        save=True,
        augments=None,
        test_set=None,
        seed=42):
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Get device (CPU/GPU) automatically
    device = get_device()
    predictor = predictor.to(device)
    
    model_module = importlib.import_module('.' + model_name, 'eagle.models')

    # Create the full_gcn_with_metrics folder in the project results directory
    project_root = pathlib.Path(__file__).parent.parent.parent
    full_gcn_dir = project_root / "results" / "full_gcn_with_metrics"
    full_gcn_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the full_gcn_with_metrics directory for all outputs
    outdir = full_gcn_dir / model_name / metric / device_name / predictor_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Initialize metrics collector with experiment ID
    experiment_id = f"{model_name}_{predictor_name}_{int(time.time())}"
    if exp_name:
        experiment_id = f"{exp_name}_{experiment_id}"
    
    metrics_collector = MetricsCollector(experiment_id=experiment_id)
    total_training_start_time = time.time()

    # Compute model complexity
    if training_set:
        try:
            # Get a sample for complexity analysis
            sample_g, sample_latency = training_set[0]
            sample_adjacency, sample_features, _, _ = infer.prepare_tensors(
                [sample_g], [sample_latency], model_module, predictor.binary_classifier, False, augments=augments
            )
            
            # Move sample to device for complexity analysis
            sample_adjacency = sample_adjacency.to(device)
            sample_features = sample_features.to(device)
            
            complexity_info = metrics_collector.complexity_analyzer.compute_model_complexity(
                predictor, (sample_adjacency, sample_features)
            )
            metrics_collector.set_model_complexity(complexity_info)
            print(f"Model complexity: {complexity_info}")
        except Exception as e:
            print(f"Warning: Could not compute model complexity: {e}")

    if tensorboard:
        import torch.utils.tensorboard as tb
        handler = tb.SummaryWriter(f'tensorboard/{exp_name}')

    if reset_last:
        predictor.reset_last()

    if optim_name == 'adamw':
        optimizer = optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optim_name}')

    if lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, threshold=0.01)
    elif lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    else:
        raise ValueError(f'Unknown lr scheduler: {lr_scheduler}')

    if not predictor.binary_classifier:
        criterion = torch.nn.L1Loss(reduction='sum')
    else:
        if predictor.binary_classifier == 'oneway' or predictor.binary_classifier == 'oneway-hard':
            criterion = torch.nn.BCELoss(reduction='sum')
        else:
            criterion = torch.nn.KLDivLoss(reduction='sum')

    es = utils.EarlyStopping(mode='min', patience=es_patience)

    if predictor.binary_classifier:
        training_set = utils.ProductList(training_set)
        def collate_fn(batch):
            return [[e[0] for e in pair] for pair in batch], [[e[1] for e in pair] for pair in batch]
    else:
        def collate_fn(batch):
            return [e[0] for e in batch], [e[1] for e in batch]

    training_data = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    data = validation_set

    train_corrects = [0, 0, 0, 0]
    test_corrects = [0, 0, 0, 0]
    best_accuracies = [0, 0, 0, 0]
    best_epochs = [0, 0, 0, 0]
    leeways = [0.01, 0.05, 0.1, 0.2] # +-% Accuracies
    lowest_loss = None

    if warmup:
        print(f'Warming up the last layer for {warmup} epochs')
        warmup_opt = optim.AdamW(predictor.final_params(), lr=learning_rate, weight_decay=0)
        for warmup_epoch in range(warmup):
            print(f"Warmup Epoch: {warmup_epoch}")
            for g, latency in training_data:
                loss = _train(model_module, predictor, g, latency, warmup_opt, criterion, device, augments=augments)

    for epoch_no in range(epochs):
        print(f"Epoch: {epoch_no}")
        
        # Start epoch metrics collection
        metrics_collector.start_epoch()

        # Training phase
        epoch_train_loss = 0.0
        train_targets = []
        train_predictions = []
        
        for g, latency in training_data:
            loss = _train(model_module, predictor, g, latency, optimizer, criterion, device, augments=augments)
            epoch_train_loss += loss.item()

        # Evaluate on training set
        if not predictor.binary_classifier:
            for g, latency in training_set:
                corrects, loss, (target, pred) = _test(model_module, predictor, g, latency, leeways, criterion, device, augments=augments)
                for i, c in enumerate(corrects):
                    train_corrects[i] += c
                epoch_train_loss += loss.item()
                if target is not None and pred is not None:
                    train_targets.append(target)
                    train_predictions.append(pred)
        else:
            for g, latency in training_data:
                _, loss, _ = _test(model_module, predictor, g, latency, None, criterion, device, augments=augments)
                epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(training_set) if training_set else 0
        
        # Record training metrics
        if train_targets and train_predictions:
            metrics_collector.record_train_metrics(avg_train_loss, train_targets, train_predictions)

        # Validation phase
        epoch_val_loss = 0.0
        val_targets = []
        val_predictions = []
        
        if not predictor.binary_classifier:
            # Measure inference time for validation
            inference_start_time = time.time()
            
            for g, latency in validation_set:
                corrects, loss, (target, pred) = _test(model_module, predictor, g, latency, leeways, criterion, device, augments=augments)
                for i, c in enumerate(corrects):
                    test_corrects[i] += c
                epoch_val_loss += loss.item()
                if target is not None and pred is not None:
                    val_targets.append(target)
                    val_predictions.append(pred)
            
            inference_time = time.time() - inference_start_time
            metrics_collector.record_inference_time(inference_time)
            
            avg_val_loss = epoch_val_loss / len(validation_set) if validation_set else 0

            if validation_set:
                current_accuracies = [test_correct / len(validation_set) for test_correct in test_corrects]
            else:
                current_accuracies = [0, 0, 0, 0]
            print(f'Average loss of validation set {epoch_no}: {avg_val_loss}')
            
            metrics_collector.record_val_metrics(avg_val_loss, val_targets, val_predictions, epoch_no)

            for i, best_accuracy in enumerate(best_accuracies):
                if current_accuracies[i] >= best_accuracy:
                    best_accuracies[i] = current_accuracies[i]
                    best_epochs[i] = epoch_no
        else:
            avg_val_loss = epoch_train_loss

        # End epoch metrics collection
        metrics_collector.end_epoch()

        # Print metrics
        if not predictor.binary_classifier and training_set:
            train_accuracies = [train_correct / len(training_set) for train_correct in train_corrects]
            print(f'Top +-{leeways} Accuracy of train set for epoch {epoch_no}: {train_accuracies} ')
            if validation_set:
                print(f'Top +-{leeways} Accuracy of validation set for epoch {epoch_no}: {current_accuracies}')
                print(f'[best: {best_accuracies} @ epoch {best_epochs}]')
            
            if (metrics_collector.train_mae and epoch_no < len(metrics_collector.train_mae) and 
                metrics_collector.val_mae and epoch_no < len(metrics_collector.val_mae)):
                print(f'Train MAE: {metrics_collector.train_mae[-1]:.6f}, Val MAE: {metrics_collector.val_mae[-1]:.6f}')
                print(f'Train R²: {metrics_collector.train_r2[-1]:.4f}, Val R²: {metrics_collector.val_r2[-1]:.4f}')
        
        print(f'Average loss of training set {epoch_no}: {avg_train_loss:.6f}')
        if metrics_collector.epoch_times:
            print(f'Epoch time: {metrics_collector.epoch_times[-1]:.2f}s')
        if PSUTIL_AVAILABLE and metrics_collector.memory_usage:
            print(f'Memory usage: {metrics_collector.memory_usage[-1]:.2f} MB')
            print(f'CPU usage: {metrics_collector.cpu_usage[-1]:.2f}%')
        if PYRAPL_AVAILABLE and metrics_collector.energy_consumption:
            print(f'Energy consumption: {metrics_collector.energy_consumption[-1]:.2f} J')
        if metrics_collector.inference_times:
            print(f'Inference time: {metrics_collector.inference_times[-1]:.4f}s')

        # Model checkpointing
        if lowest_loss is None or avg_val_loss < lowest_loss:
            lowest_loss = avg_val_loss
            best_predictor_weight = predictor.state_dict()
            if save:
                torch.save(best_predictor_weight, outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt'))
            print(f'Lowest val_loss: {avg_val_loss:.6f}... Predictor model saved.')

        # Learning rate scheduling
        if lr_scheduler == 'plateau':
            if epoch_no > 20:
                scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Early stopping
        if epoch_no > 20:
            if es.step(avg_val_loss):
                print('Early stopping criterion is met, stop training now.')
                break

        # Reset counters for next epoch
        train_corrects = [0, 0, 0, 0]
        test_corrects = [0, 0, 0, 0]

        # Tensorboard logging
        if tensorboard:
            handler.add_scalar('loss/training', avg_train_loss, epoch_no)
            handler.add_scalar('loss/validation', avg_val_loss, epoch_no)
            if not predictor.binary_classifier and training_set and validation_set:
                handler.add_scalar('accuracy_1/training', train_accuracies[0], epoch_no)
                handler.add_scalar('accuracy_1/validation', current_accuracies[0], epoch_no)
                handler.add_scalar('accuracy_5/training', train_accuracies[1], epoch_no)
                handler.add_scalar('accuracy_5/validation', current_accuracies[1], epoch_no)
                handler.add_scalar('accuracy_10/training', train_accuracies[2], epoch_no)
                handler.add_scalar('accuracy_10/validation', current_accuracies[2], epoch_no)
                handler.add_scalar('accuracy_20/training', train_accuracies[3], epoch_no)
                handler.add_scalar('accuracy_20/validation', current_accuracies[3], epoch_no)
                if (metrics_collector.train_mae and epoch_no < len(metrics_collector.train_mae) and
                    metrics_collector.val_mae and epoch_no < len(metrics_collector.val_mae)):
                    handler.add_scalar('metrics/mae_train', metrics_collector.train_mae[-1], epoch_no)
                    handler.add_scalar('metrics/mae_val', metrics_collector.val_mae[-1], epoch_no)
                    handler.add_scalar('metrics/r2_train', metrics_collector.train_r2[-1], epoch_no)
                    handler.add_scalar('metrics/r2_val', metrics_collector.val_r2[-1], epoch_no)
            if metrics_collector.epoch_times:
                handler.add_scalar('system/epoch_time', metrics_collector.epoch_times[-1], epoch_no)
            if PSUTIL_AVAILABLE and metrics_collector.memory_usage:
                handler.add_scalar('system/memory_usage', metrics_collector.memory_usage[-1], epoch_no)
                handler.add_scalar('system/cpu_usage', metrics_collector.cpu_usage[-1], epoch_no)
            if PYRAPL_AVAILABLE and metrics_collector.energy_consumption:
                handler.add_scalar('system/energy_consumption', metrics_collector.energy_consumption[-1], epoch_no)
            if metrics_collector.inference_times:
                handler.add_scalar('system/inference_time', metrics_collector.inference_times[-1], epoch_no)

    # End of training
    total_training_time = time.time() - total_training_start_time
    
    if tensorboard:
        handler.close()
    
    if save:
        torch.save(best_predictor_weight, outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt'))

    # Evaluate on test set if provided
    if test_set is not None and not predictor.binary_classifier:
        print("\n=== FINAL TEST SET EVALUATION ===")
        test_targets = []
        test_predictions = []
        test_loss = 0.0
        
        inference_start_time = time.time()
        for g, latency in test_set:
            _, loss, (target, pred) = _test(model_module, predictor, g, latency, leeways, criterion, device, augments=augments)
            test_loss += loss.item()
            if target is not None and pred is not None:
                test_targets.append(target)
                test_predictions.append(pred)
        
        inference_time = time.time() - inference_start_time
        metrics_collector.record_inference_time(inference_time)
        
        if test_targets and test_predictions:
            metrics_collector.record_test_metrics(test_targets, test_predictions)
            print(f"Test set evaluation completed:")
            print(f"Test MAE: {metrics_collector.test_mae:.6f}")
            print(f"Test R²: {metrics_collector.test_r2:.4f}")
            print(f"Test Accuracy 1%: {metrics_collector.test_accuracies[0]:.4f}")
            print(f"Test Accuracy 5%: {metrics_collector.test_accuracies[1]:.4f}")
            print(f"Test Accuracy 10%: {metrics_collector.test_accuracies[2]:.4f}")
            print(f"Test Accuracy 20%: {metrics_collector.test_accuracies[3]:.4f}")

    # Save final results
    cfg = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size
    }
    
    results = metrics_collector.get_final_results(cfg, model_name, total_training_time)
    results_filepath = metrics_collector.save_results(results, outdir, model_name, exp_name)
    
    # Create charts
    charts_dir = metrics_collector.create_charts(results, outdir, model_name, exp_name)
    
    print("Training finished!")
    print(f"Results saved to: {results_filepath}")
    if charts_dir:
        print(f"Charts saved to: {charts_dir}")
    
    # Print summary of final results
    print("\n=== TRAINING SUMMARY ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Best validation epoch: {results['best_epoch']}")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Final validation MAE: {results['mae']:.6f}")
    print(f"Final validation R²: {results['r2']:.4f}")
    print(f"Validation Accuracy 1%: {results['accuracy_1%']:.4f}")
    print(f"Validation Accuracy 5%: {results['accuracy_5%']:.4f}")
    print(f"Validation Accuracy 10%: {results['accuracy_10%']:.4f}")
    print(f"Validation Accuracy 20%: {results['accuracy_20%']:.4f}")
    
    if test_set is not None:
        print(f"Test MAE: {results['test_mae']:.6f}")
        print(f"Test R²: {results['test_r2']:.4f}")
        print(f"Test Accuracy 1%: {results['test_accuracy_1%']:.4f}")
        print(f"Test Accuracy 5%: {results['test_accuracy_5%']:.4f}")
        print(f"Test Accuracy 10%: {results['test_accuracy_10%']:.4f}")
        print(f"Test Accuracy 20%: {results['test_accuracy_20%']:.4f}")
    
    # Model complexity summary
    if metrics_collector.model_complexity:
        print(f"Total parameters: {metrics_collector.model_complexity.get('total_parameters', 0):,}")
        print(f"Trainable parameters: {metrics_collector.model_complexity.get('trainable_parameters', 0):,}")
        print(f"Estimated FLOPS: {metrics_collector.model_complexity.get('estimated_flops', 0):,}")
    
    # System metrics summary
    if PSUTIL_AVAILABLE:
        print(f"Average memory usage: {results['average_memory_usage_mb']:.2f} MB")
        print(f"Average CPU usage: {results['average_cpu_percent']:.2f}%")
    if PYRAPL_AVAILABLE:
        print(f"Total energy consumption: {results['total_energy_joules']:.2f} J")
        print(f"Average energy per epoch: {results['average_energy_per_epoch_joules']:.2f} J")
    
    print(f"Average inference time: {results['average_inference_time_seconds']:.4f}s")
    print(f"Training completed with seed: {seed}")

    predictor.load_state_dict(best_predictor_weight)
    return predictor, results_filepath


def predict(testing_data,
        outdir,
        device_name,
        model_name,
        metric,
        predictor_name,
        predictor,
        log=False,
        exp_name=None,
        load=False,
        iteration=None,
        explored_models=None,
        valid_pts=None,
        use_fast=True,
        augments=None):
    
    # Get device automatically
    device = get_device()
    predictor = predictor.to(device)
    
    model_module = importlib.import_module('.' + model_name, 'eagle.models')

    # Create the full_gcn_with_metrics folder in the project results directory
    project_root = pathlib.Path(__file__).parent.parent.parent
    full_gcn_dir = project_root / "results" / "full_gcn_with_metrics"
    full_gcn_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the full_gcn_with_metrics directory for all outputs
    if load or log:
        outdir = full_gcn_dir / model_name / metric / device_name / predictor_name
        if log:
            outdir.mkdir(parents=True, exist_ok=True)

    if load and predictor_name != 'random':
        predictor.load_state_dict(torch.load(outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt')))
        print('Predictor imported.')

    criterion = torch.nn.L1Loss()

    test_corrects = [0, 0, 0, 0]
    leeways = [0.01, 0.05, 0.1, 0.2]

    log_file = None
    if log:
        log_filename = 'log.txt' if exp_name is None else f'log_{exp_name}.txt'
        if iteration is not None:
            log_filename = f'iter{iteration}_' + log_filename

        log_file = outdir / log_filename
        sep = False
        if log_file.exists():
            sep = True
        log_file = log_file.open('a')
        if sep:
            log_file.write('===\n')

    if predictor_name == 'random':
        print('Producing random ordering of the dataset...')
        predicted = []
        perm = np.random.permutation(len(testing_data))
        for idx, (point, gt_value) in enumerate(testing_data):
            predicted_value = perm[idx]
            predicted.append(predicted_value)
            if log:
                log_file.write(f'{gt_value} {predicted_value} {point}\n')

    elif not predictor.binary_classifier:
        predicted = []
        test_loss = 0
        test_targets = []
        test_predictions = []
        
        for g, latency in testing_data:
            corrects, loss, values = _test(model_module, predictor, g, latency, leeways, criterion, device, log_file, augments)
            for i, c in enumerate(corrects):
                test_corrects[i] += c

            test_loss += loss
            predicted.append(values[1])
            test_targets.append(values[0])
            test_predictions.append(values[1])

        current_accuracies = [test_correct / len(testing_data) for test_correct in test_corrects]
        avg_loss = test_loss / len(testing_data)

        print(f'Top +-{leeways} Accuracy of test set: {current_accuracies}')
        print(f'Average loss of test set: {avg_loss}')
        
        # Calculate and print additional test metrics
        if test_targets and test_predictions:
            test_mae = mean_absolute_error(test_targets, test_predictions)
            test_r2 = r2_score(test_targets, test_predictions)
            print(f'Test MAE: {test_mae:.6f}')
            print(f'Test R²: {test_r2:.4f}')
    else:
        torch.set_grad_enabled(False)
        predictor.eval()

        if use_fast:
            print(f'Precomputing embeddings for {len(testing_data)} graphs')
            precomputed = infer.precompute_embeddings(model_module, predictor, testing_data, 1024, augments=augments)
            print('Done')

        total = 0
        correct = 0
        skipped = 0
        def predictor_compare(v1, v2):
            nonlocal total
            nonlocal correct
            nonlocal skipped
            total += 1
            if valid_pts is not None and v1[0] not in valid_pts:
                skipped += 1
                return -1
            if valid_pts is not None and v2[0] not in valid_pts:
                skipped += 1
                return 1
            latencies = [v1[1], v2[1]]
            if use_fast:
                result = infer.precomputed_forward(predictor, [v1[2], v2[2]], precomputed)
            else:
                gs = [v1[0], v2[0]]
                adjacency, features, _, aug = infer.prepare_tensors([gs], None, model_module, predictor.binary_classifier, False, augments=augments)
                if augments is not None:
                    result = predictor(adjacency, features, aug)
                else:
                    result = predictor(adjacency, features)
            if predictor.binary_classifier == 'oneway' or predictor.binary_classifier == 'oneway-hard':
                v1_better = result[0][0].cpu().item() - 0.5
                if latencies[0] > latencies[1]:
                    if v1_better > 0:
                        correct += 1
                elif v1_better < 0:
                    correct += 1

                return v1_better
            else:
                rv1, rv2 = result[0][0].cpu().item(), result[0][1].cpu().item()
                if latencies[0] > latencies[1]:
                    if rv1 > rv2:
                        correct += 1
                elif rv1 < rv2:
                    correct += 1

                return rv1 - rv2

        if use_fast:
            predictor.cpu()
            test_data_with_indices = [(*v, idx) for idx, v in enumerate(testing_data)]
            sorted_values = sorted(test_data_with_indices, key=functools.cmp_to_key(predictor_compare))
            sorted_values = { pt: (gt,idx) for idx,(pt,gt,_) in enumerate(sorted_values) }
        else:
            sorted_values = sorted(testing_data, key=functools.cmp_to_key(predictor_compare))
            sorted_values = { pt: (gt,idx) for idx,(pt,gt) in enumerate(sorted_values) }

        predicted = []
        for p, v in testing_data:
            r = sorted_values[p][1]
            predicted.append(r)
            if log:
                log_file.write(f'{v} {r} {p}\n')

        predictor.train()
        torch.set_grad_enabled(True)

    if log:
        log_file.write('---\n')
        explored_models = explored_models or []
        for p,v in explored_models:
            log_file.write(f'{p}\n')
        log_file.write('---\n')
        if predictor_name == 'random':
            pass
        elif not predictor.binary_classifier:
            log_file.write(f'{avg_loss}\n{current_accuracies}\n')
        else:
            log_file.write(f'{correct}/{total} predictions correct\n')
            log_file.write(f'{skipped}/{total} predictions skipped\n')
        log_file.close()

    return predicted


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model family to run, should be a name of one of the packages under eagle.models')
    parser.add_argument('--device', type=str, required=True, help='Device on which the models will be run, should be a name of one of the packages under eagle.device_runner')
    parser.add_argument('--metric', type=str, default='latency', help='Metric to measure. Default: latency.')
    parser.add_argument('--predictor', type=str, required=True, help='Predictor to train, should a name of one of the packages under eagle.predictors')
    parser.add_argument('--measurement', type=str, required=True, default=None, help='Measurement file for device')
    parser.add_argument('--cfg', type=str, default=None, help='Configuration file for device and model packages')
    parser.add_argument('--expdir', type=str, default='results', help='Folder in which the results of measurements will be saved. Default: results')
    parser.add_argument('--process', action='store_true', help='Process measurements - use this if the measurements are not already processed')
    parser.add_argument('--multiple_files', action='store_true', help='Combine results from multiple files - use this if the measurements are not already combined')
    parser.add_argument('--transfer', default=None, help='Perform transfer learning from a previously trained model - the argument should point to the checkpoint to load')
    parser.add_argument('--load', default=None, help='Checkpoint to load')
    parser.add_argument('--warmup', default=0, type=int, help='Number of warmup epochs for the last layer')
    parser.add_argument('--foresight_warmup', type=str, help='Path to the dataset containing foresight metrics which will be used to warmup the predictor during iterative training')
    parser.add_argument('--foresight_simple', action='store_true', help='Do not train the predictor when doing foresight warmup, instead simply rank models with foresight scores directly')
    parser.add_argument('--foresight_augment', type=str, nargs='+', default=[], help='Path to foresight metrics, if set they will be passed to the predictor together with each model')
    parser.add_argument('--prediction_only', action='store_true', help='Run prediction with a pretrained predictor')
    parser.add_argument('--exp', default=None, help='Optional experiment name, used when saving the predictor to distinguish between different configurations')
    parser.add_argument('--uid', default=None, type=int, help='UID to distinguish between different concurrent runs')
    parser.add_argument('--log', action='store_true', help='Log prediction on test dataset together with ground truth')
    parser.add_argument('--tensorboard', action='store_true', help='Log training data for visualization in tensorboard')
    parser.add_argument('--torch_seed', type=int, default=None, help='Fixed seed to use with torch.random')
    parser.add_argument('--quiet', action='store_true', help='Suppress standard output')
    parser.add_argument('--iter', type=int, default=0, help='Number of iterations when using iterative search')
    parser.add_argument('--save', action='store_true', help='Save the best predictor')
    parser.add_argument('--eval', action='store_true', help='Eval model only, do not train (use with --transfer to eval pretrained model)')
    parser.add_argument('--lat_limit', type=float, default=None, help='Latency limit to prune the search space (requires --transfer to point to the latency predictor)')
    parser.add_argument('--sample_best', action='store_true')
    parser.add_argument('--sample_best2', action='store_true')
    parser.add_argument('--reset_last', action='store_true', help='Reset last layer (only applicable if checkpoint is loaded)')

    parser.add_argument('--leave_one_out', type=str)
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()

    if args.uid is not None:
        if args.exp is None:
            args.exp = str(args.uid)
        else:
            args.exp += f'_{args.uid}'

    with contextlib.ExitStack() as es:
        if args.quiet:
            f = es.enter_context(open(os.devnull, 'w'))
            es.enter_context(contextlib.redirect_stdout(f))

        extra_args = {}
        if args.cfg:
            import yaml
            with open(args.cfg, 'r') as f:
                extra_args = yaml.load(f, Loader=yaml.Loader)

        if args.transfer:
            if not args.load:
                raise ValueError('Both --load and --transfer are set, please use only one. Note: "--transfer X" is the same as "--load X --reset_last"')

            args.load = args.transfer
            args.reset_last = True

        if args.predictor == 'random':
            predictor = None
        else:
            predictor = infer.get_predictor(args.predictor, predictor_args=extra_args.get('predictor'), checkpoint=args.load, ignore_last=args.reset_last, augment=len(args.foresight_augment))
        lat_predictor = None
        if args.lat_limit:
            if not args.transfer:
                raise ValueError('--lat_limit requires --transfer')

            lat_predictor_args = extra_args.get('predictor').copy()
            lat_predictor_args.pop('binary_classifier')
            lat_predictor = infer.get_predictor(args.predictor, predictor_args=lat_predictor_args, checkpoint=args.load, ignore_last=False)

        # if args.predictor != 'random':
        #     if torch.cuda.is_available():
        #         predictor.cuda()
        #         if lat_predictor:
        #             lat_predictor.cuda()
            # else:
                # raise RuntimeError('No GPU!')

        if args.model == 'darts':
            dataset_args = extra_args.get('dataset', {})
            dataset_file = dataset_args.pop('dataset_file', None)
            if dataset_file:
                dataset_file = pathlib.Path(args.expdir) / args.model / args.metric / args.device / dataset_file
            dataset = dataset_mod.DartsDataset(args.measurement,
                                    dataset_file=dataset_file,
                                    **extra_args.get('dataset', {}))
        else:
            dataset = dataset_mod.EagleDataset(args.measurement,
                                    args.process,
                                    args.multiple_files,
                                    **extra_args.get('dataset', {}),
                                    lat_limit=args.lat_limit,
                                    lat_predictor=lat_predictor,
                                    model_module=importlib.import_module('.' + args.model, 'eagle.models'))

            if args.foresight_warmup:
                if not args.iter:
                    raise ValueError('Foresight warmup requires iterative training!')
                if args.foresight_augment:
                    raise ValueError('Foresigh augment is incompatible with foresight warmup')

                foresight_dataset = dataset_mod.EagleDataset(args.foresight_warmup,
                    args.process,
                    args.multiple_files,
                    **extra_args.get('foresight', {}).get('dataset', {}),
                    lat_limit=args.lat_limit,
                    lat_predictor=lat_predictor,
                    model_module=importlib.import_module('.' + args.model, 'eagle.models'))

            if args.foresight_augment:
                print(f'Using {len(args.foresight_augment)} foresight metric(s) to augment graph embeddings')
                augments = []
                for aug in args.foresight_augment:
                    with open(aug, 'rb') as f:
                        d = pickle.load(f)
                        augments.append(d)
            else:
                augments = None

        explored_models = dataset.train_set
        if not args.eval:
            if args.iter:
                if args.foresight_warmup:
                    if not args.foresight_simple:
                        print(f'Warming up predictor using foresight dataset {args.foresight_warmup!r}')
                        foresight_train_args = extra_args.get('foresight', {}).get('training', {})
                        train(foresight_dataset.train_set,
                            foresight_dataset.valid_set,
                            args.expdir,
                            args.device,
                            args.model,
                            args.metric,
                            args.predictor,
                            predictor,
                            args.tensorboard,
                            **foresight_train_args,
                            exp_name=args.exp,
                            reset_last=args.reset_last,
                            warmup=args.warmup,
                            save=False,
                            augments=augments)
                    else:
                        print(f'Sorting models using foresight metrics from: {args.foresight_warmup!r}')

                train_args = extra_args.get('training', {})

                target_batch = train_args.pop('batch_size')
                batch_per_iter = target_batch // args.iter
                current_batch = batch_per_iter

                target_epochs = train_args.pop('epochs')
                epochs_per_iter = target_epochs // args.iter
                current_epochs = epochs_per_iter

                points_per_iter = len(dataset.train_set) // args.iter
                candidates = list(dataset.dataset)

                if not args.foresight_warmup:
                    train_set = dataset_mod.select_random(candidates, points_per_iter)
                else:
                    train_set = []

                for i in range(args.iter):
                    print('Iteration', i)

                    if i or args.foresight_warmup:
                        # update training set
                        if i or not args.foresight_simple:
                            scores = predict(candidates, args.expdir, args.device, args.model, args.metric, args.predictor, predictor, log=False, exp_name=args.exp, load=False, augments=augments)
                        else:
                            scores = [p[1] for p in foresight_dataset.dataset]
                        if args.sample_best or args.sample_best2:
                            if not args.sample_best2:
                                median_score = statistics.median(scores)
                                candidates = [pt for pt, score in zip(candidates, scores) if score > median_score]
                            best_candidates = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
                            added = 0
                            for candidate, score in best_candidates:
                                if added == points_per_iter//2:
                                    break
                                if candidate in train_set:
                                    continue
                                train_set.append(candidate)
                                added += 1

                            if args.sample_best2:
                                random_th = best_candidates[len(scores) // (2**(i or 1))][1]
                                random_candidates = [pt for pt, score in zip(candidates, scores) if score > random_th]
                                selected_candidates = dataset_mod.select_random(random_candidates, points_per_iter//2, current=train_set)
                            else:
                                selected_candidates = dataset_mod.select_random(candidates, points_per_iter//2, current=train_set)
                            train_set.extend(selected_candidates)
                        else:
                            median_score = statistics.median(scores)
                            candidates = [pt for pt, score in zip(candidates, scores) if score > median_score]
                            sampled = dataset_mod.select_random(candidates, points_per_iter, current=train_set)
                            train_set.extend(sampled)

                    print('Number of candidate points:', len(candidates))
                    print('Number of training points:', len(train_set))
                    print('Batch size:', current_batch)
                    print('Number of epochs:', current_epochs)

                    train(train_set,
                        train_set,
                        args.expdir,
                        args.device,
                        args.model,
                        args.metric,
                        args.predictor,
                        predictor,
                        args.tensorboard,
                        **train_args,
                        batch_size=current_batch,
                        epochs=current_epochs,
                        exp_name=args.exp,
                        reset_last=args.reset_last and not i and not args.foresight_warmup,
                        warmup=args.warmup if (not i and not args.foresight_warmup) else 0,
                        save=args.save and i + 1 == args.iter,
                        augments=augments)

                    current_batch += batch_per_iter
                    current_epochs += epochs_per_iter
                    explored_models = train_set
            else:
                if not dataset.train_set:
                    raise ValueError('Training set is empty!')
                train(dataset.train_set,
                    dataset.valid_set,
                    args.expdir,
                    args.device,
                    args.model,
                    args.metric,
                    args.predictor,
                    predictor,
                    args.tensorboard,
                    **extra_args.get('training', {}),
                    exp_name=args.exp,
                    reset_last=args.reset_last,
                    warmup=args.warmup,
                    save=args.save,
                    augments=augments,
                    test_set=dataset.full_dataset)  # Pass test set for final evaluation

        predict(dataset.full_dataset,
            args.expdir,
            args.device,
            args.model,
            args.metric,
            args.predictor,
            predictor,
            args.log,
            exp_name=args.exp,
            load=False,
            explored_models=explored_models,
            valid_pts=dataset.valid_pts,
            augments=augments)
        



# import os
# import pickle
# import pathlib
# import argparse
# import importlib
# import functools
# import contextlib
# import statistics
# import time
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, r2_score
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from . import utils
# from . import infer
# from . import dataset as dataset_mod

# # Import system monitoring libraries
# try:
#     import psutil
#     PSUTIL_AVAILABLE = True
# except ImportError:
#     PSUTIL_AVAILABLE = False
#     print("Warning: psutil not available. CPU and memory metrics will not be collected.")
# try:
#     import pyRAPL
#     PYRAPL_AVAILABLE = True
# except ImportError:
#     PYRAPL_AVAILABLE = False
#     print("Warning: pyRAPL not available. Energy consumption metrics will not be collected.")


# class ModelComplexityAnalyzer:
#     """Class to analyze model complexity including FLOPS and parameters"""
#     def __init__(self):
#         self.model_flops = {}
#         self.model_parameters = {}
    
#     def compute_model_complexity(self, model, sample_input):
#         """Compute FLOPS and parameters for a model"""
#         try:
#             # Compute number of parameters
#             total_params = sum(p.numel() for p in model.parameters())
#             self.model_parameters['total'] = total_params
#             self.model_parameters['trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
#             # Estimate FLOPS (this is a simplified estimation)
#             # For more accurate FLOPS calculation, consider using torchprofile or thop
#             flops_estimate = self.estimate_flops(model, sample_input)
#             self.model_flops['estimated'] = flops_estimate
            
#             return {
#                 'total_parameters': total_params,
#                 'trainable_parameters': self.model_parameters['trainable'],
#                 'estimated_flops': flops_estimate
#             }
#         except Exception as e:
#             print(f"Warning: Could not compute model complexity: {e}")
#             return {
#                 'total_parameters': 0,
#                 'trainable_parameters': 0,
#                 'estimated_flops': 0
#             }
    
#     def estimate_flops(self, model, sample_input):
#         """Estimate FLOPS for the model (simplified version)"""
#         # This is a basic estimation - for accurate FLOPS, use specialized libraries
#         try:
#             if hasattr(model, 'estimate_flops'):
#                 return model.estimate_flops(sample_input)
            
#             # Simple estimation based on parameters and operations
#             total_flops = 0
#             for module in model.modules():
#                 if isinstance(module, nn.Linear):
#                     # FLOPs for linear layer: 2 * input_size * output_size
#                     if hasattr(module, 'weight'):
#                         in_features = module.in_features
#                         out_features = module.out_features
#                         total_flops += 2 * in_features * out_features
#                 elif isinstance(module, nn.Conv2d):
#                     # FLOPs for conv2d: 2 * kernel_size * kernel_size * in_channels * out_channels * output_height * output_width
#                     if hasattr(module, 'weight'):
#                         kernel_size = module.kernel_size[0] * module.kernel_size[1]
#                         in_channels = module.in_channels
#                         out_channels = module.out_channels
#                         # Estimate output size (this is simplified)
#                         output_size = 112  # typical for many architectures
#                         total_flops += 2 * kernel_size * in_channels * out_channels * output_size * output_size
            
#             return total_flops
#         except:
#             return 0


# class MetricsCollector:
#     """Class to collect and compute performance and system metrics during training"""
    
#     def __init__(self, experiment_id=None):
#         self.experiment_id = experiment_id or f"exp_{int(time.time())}"
#         self.experiment_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
#         # Training metrics
#         self.train_losses = []
#         self.val_losses = []
#         self.train_mae = []
#         self.val_mae = []
#         self.train_r2 = []
#         self.val_r2 = []
#         self.train_accuracies = []  # [acc1, acc5, acc10, acc20] for each epoch
#         self.val_accuracies = []    # [acc1, acc5, acc10, acc20] for each epoch
#         self.epoch_times = []
        
#         # System metrics
#         self.memory_usage = []
#         self.cpu_usage = []
#         self.energy_consumption = []
        
#         # Model complexity metrics
#         self.model_complexity = {}
#         self.inference_times = []
        
#         # For final results - store BEST epoch data, not accumulated
#         self.best_val_targets = []
#         self.best_val_predictions = []
#         self.best_val_loss = float('inf')
#         self.best_epoch = -1
        
#         # Test set metrics
#         self.test_targets = []
#         self.test_predictions = []
#         self.test_mae = 0.0
#         self.test_r2 = 0.0
#         self.test_accuracies = [0.0, 0.0, 0.0, 0.0]
        
#         # Energy measurement setup
#         if PYRAPL_AVAILABLE:
#             pyRAPL.setup()
#             self.energy_meter = pyRAPL.Measurement('training')
        
#         # Initialize complexity analyzer
#         self.complexity_analyzer = ModelComplexityAnalyzer()
    
#     def set_model_complexity(self, complexity_info):
#         """Set model complexity information"""
#         self.model_complexity = complexity_info
    
#     def start_epoch(self):
#         """Start measurement for a new epoch"""
#         self.epoch_start_time = time.time()
#         if PYRAPL_AVAILABLE:
#             self.energy_meter.begin()
        
#         # Record initial system stats
#         if PSUTIL_AVAILABLE:
#             self.initial_memory = psutil.virtual_memory().used
#             self.initial_cpu = psutil.cpu_percent(interval=None)
    
#     def end_epoch(self):
#         """End measurement for current epoch"""
#         epoch_time = time.time() - self.epoch_start_time
#         self.epoch_times.append(epoch_time)
        
#         # Record system metrics
#         if PSUTIL_AVAILABLE:
#             # Memory usage in MB
#             current_memory = psutil.virtual_memory().used
#             memory_used_mb = (current_memory - self.initial_memory) / (1024 * 1024)
#             self.memory_usage.append(max(0, memory_used_mb))  # Avoid negative values
            
#             # CPU usage
#             current_cpu = psutil.cpu_percent(interval=None)
#             self.cpu_usage.append((self.initial_cpu + current_cpu) / 2)
        
#         # Energy consumption
#         if PYRAPL_AVAILABLE:
#             self.energy_meter.end()
#             energy_joules = sum(self.energy_meter.result.energy)
#             self.energy_consumption.append(energy_joules)
    
#     def record_inference_time(self, inference_time):
#         """Record inference time for a batch"""
#         self.inference_times.append(inference_time)
    
#     def compute_regression_metrics(self, targets, predictions, leeways=[0.01, 0.05, 0.1, 0.2]):
#         """Compute regression metrics for given targets and predictions with numerical stability"""
#         if not targets or not predictions:
#             return 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], []
            
#         # Convert to numpy arrays with proper handling
#         targets_np = np.array(targets, dtype=np.float64)
#         predictions_np = np.array(predictions, dtype=np.float64)
        
#         # Check for valid data range
#         if len(targets_np) < 2 or np.std(targets_np) < 1e-12:
#             # If all targets are the same, R² is undefined - return 0 instead of -1
#             mae = mean_absolute_error(targets_np, predictions_np)
            
#             # Calculate accuracies
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 relative_errors = np.abs((predictions_np - targets_np) / np.abs(targets_np))
#                 relative_errors = np.nan_to_num(relative_errors, nan=1.0, posinf=1.0, neginf=1.0)
            
#             accuracies = []
#             for leeway in leeways:
#                 accuracy = np.mean(relative_errors <= leeway)
#                 accuracies.append(float(accuracy))
            
#             return float(mae), 0.0, accuracies, relative_errors.tolist()
        
#         # MAE calculation
#         mae = mean_absolute_error(targets_np, predictions_np)
        
#         # **FIXED: Proper R² calculation**
#         try:
#             r2 = r2_score(targets_np, predictions_np)
#             # Ensure R² is within reasonable bounds
#             r2 = max(min(r2, 1.0), -1.0)
#         except:
#             r2 = 0.0
        
#         # Accuracy within error bounds with numerical stability
#         with np.errstate(divide='ignore', invalid='ignore'):
#             relative_errors = np.abs((predictions_np - targets_np) / np.abs(targets_np))
#             relative_errors = np.nan_to_num(relative_errors, nan=1.0, posinf=1.0, neginf=1.0)
        
#         accuracies = []
#         for leeway in leeways:
#             accuracy = np.mean(relative_errors <= leeway)
#             accuracies.append(float(accuracy))
        
#         return float(mae), float(r2), accuracies, relative_errors.tolist()
    
#     def record_train_metrics(self, loss, targets, predictions):
#         """Record training metrics for current epoch"""
#         self.train_losses.append(float(loss))
#         if targets and predictions:
#             mae, r2, accuracies, _ = self.compute_regression_metrics(targets, predictions)
#             self.train_mae.append(mae)
#             self.train_r2.append(r2)
#             self.train_accuracies.append(accuracies)
    
#     def record_val_metrics(self, loss, targets, predictions, epoch):
#         """Record validation metrics for current epoch and track best epoch"""
#         self.val_losses.append(float(loss))
        
#         if targets and predictions:
#             mae, r2, accuracies, _ = self.compute_regression_metrics(targets, predictions)
#             self.val_mae.append(mae)
#             self.val_r2.append(r2)
#             self.val_accuracies.append(accuracies)
            
#             # **FIXED: Only store best epoch data, not accumulate all epochs**
#             if loss < self.best_val_loss:
#                 self.best_val_loss = loss
#                 self.best_epoch = epoch
#                 self.best_val_targets = targets.copy()
#                 self.best_val_predictions = predictions.copy()
    
#     def record_test_metrics(self, targets, predictions):
#         """Record test set metrics"""
#         self.test_targets = targets.copy()
#         self.test_predictions = predictions.copy()
#         if targets and predictions:
#             self.test_mae, self.test_r2, self.test_accuracies, _ = self.compute_regression_metrics(targets, predictions)
    
#     def get_final_results(self, cfg, model_name, total_training_time):
#         """Compile final results dictionary using BEST epoch data"""
#         # **FIXED: Use best epoch data for final metrics, not accumulated data**
#         if self.best_val_targets and self.best_val_predictions:
#             final_mae, final_r2, final_accuracies, abs_errors = self.compute_regression_metrics(
#                 self.best_val_targets, self.best_val_predictions
#             )
#         else:
#             # Fallback if no validation data
#             final_mae = self.val_mae[-1] if self.val_mae else 0.0
#             final_r2 = self.val_r2[-1] if self.val_r2 else 0.0
#             final_accuracies = self.val_accuracies[-1] if self.val_accuracies else [0.0, 0.0, 0.0, 0.0]
#             abs_errors = []
        
#         # Calculate average inference time per sample
#         total_inference_time = sum(self.inference_times) if self.inference_times else sum(self.epoch_times)
#         num_samples = len(self.best_val_targets) if self.best_val_targets else 1
#         avg_inference_time = total_inference_time / max(1, num_samples)
        
#         # Calculate average system metrics
#         avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0.0
#         avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0.0
#         total_energy = sum(self.energy_consumption) if self.energy_consumption else 0.0
#         avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0.0
        
#         # Compile comprehensive results
#         results = {
#             # Experiment identification
#             'experiment_id': self.experiment_id,
#             'timestamp': self.experiment_timestamp,
#             'model_name': model_name,            
#             # Model complexity metrics
#             'model_complexity': self.model_complexity,
            
#             # Performance metrics - VALIDATION SET
#             'mae': final_mae,
#             'r2': final_r2,
#             'accuracy_1%': final_accuracies[0] if final_accuracies else 0.0,
#             'accuracy_5%': final_accuracies[1] if final_accuracies else 0.0,
#             'accuracy_10%': final_accuracies[2] if final_accuracies else 0.0,
#             'accuracy_20%': final_accuracies[3] if final_accuracies else 0.0,
            
#             # Performance metrics - TEST SET
#             'test_mae': self.test_mae,
#             'test_r2': self.test_r2,
#             'test_accuracy_1%': self.test_accuracies[0] if self.test_accuracies else 0.0,
#             'test_accuracy_5%': self.test_accuracies[1] if self.test_accuracies else 0.0,
#             'test_accuracy_10%': self.test_accuracies[2] if self.test_accuracies else 0.0,
#             'test_accuracy_20%': self.test_accuracies[3] if self.test_accuracies else 0.0,
            
#             # Time metrics
#             'total_training_time_seconds': total_training_time,
#             'average_epoch_time_seconds': avg_epoch_time,
#             'average_inference_time_seconds': avg_inference_time,
#             'total_inference_time_seconds': total_inference_time,
            
#             # System metrics
#             'average_memory_usage_mb': avg_memory,
#             'average_cpu_percent': avg_cpu,
#             'total_energy_joules': total_energy,
#             'average_energy_per_epoch_joules': total_energy / max(1, len(self.energy_consumption)),
            
#             # Training configuration
#             'epochs': cfg.get('epochs', 0),
#             'learning_rate': cfg.get('learning_rate', 0.0),
#             'batch_size': cfg.get('batch_size', 0),
#             'weight_decay': cfg.get('weight_decay', 0.0),            
#             # Training progress
#             'best_epoch': self.best_epoch,
#             'final_train_loss': self.train_losses[-1] if self.train_losses else float('nan'),
#             'final_val_loss': self.val_losses[-1] if self.val_losses else float('nan'),
#             'best_val_loss': self.best_val_loss,
#             # Data statistics
#             'target_range': [min(self.best_val_targets), max(self.best_val_targets)] if self.best_val_targets else [0.0, 0.0],
#             'prediction_range': [min(self.best_val_predictions), max(self.best_val_predictions)] if self.best_val_predictions else [0.0, 0.0],
#             'mean_absolute_error': np.mean(abs_errors) if abs_errors and len(abs_errors) > 0 else 0.0,
            
#             # Complete history for analysis
#             'train_losses': self.train_losses,
#             'val_losses': self.val_losses,
#             'train_mae': self.train_mae,
#             'val_mae': self.val_mae,
#             'train_r2': self.train_r2,
#             'val_r2': self.val_r2,
#             'train_accuracies': self.train_accuracies,
#             'val_accuracies': self.val_accuracies,
#             'epoch_times': self.epoch_times,
#             'inference_times': self.inference_times,
#             'memory_usage_history': self.memory_usage,
#             'cpu_usage_history': self.cpu_usage,
#             'energy_consumption_history': self.energy_consumption
#         }
        
#         return results
    
#     def save_comparison_data(self, results, outdir, model_name, exp_name=None):
#         """Save data in a format suitable for future comparisons"""
#         # Create comparison directory
#         comparison_dir = outdir / "comparison_data"
#         comparison_dir.mkdir(exist_ok=True)
#         # Save comprehensive results
#         filename = f"comparison_{model_name}_{self.experiment_id}"
#         if exp_name:
#             filename += f"_{exp_name}"
#         filename += ".json"
        
#         filepath = comparison_dir / filename
        
#         # Convert numpy types to Python native types for JSON serialization
#         def convert_types(obj):
#             if isinstance(obj, (np.float32, np.float64)):
#                 return float(obj)
#             elif isinstance(obj, (np.int32, np.int64)):
#                 return int(obj)
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif isinstance(obj, list):
#                 return [convert_types(x) for x in obj]
#             elif isinstance(obj, dict):
#                 return {k: convert_types(v) for k, v in obj.items()}
#             else:
#                 return obj
        
#         results_serializable = convert_types(results)
        
#         with open(filepath, 'w') as f:
#             json.dump(results_serializable, f, indent=2)
        
#         # Also save a summary CSV for easy comparison
#         self.save_comparison_summary(results, comparison_dir)
        
#         return filepath
    
#     def save_comparison_summary(self, results, comparison_dir):
#         """Save a summary CSV file for easy comparison across experiments"""
#         import csv
        
#         summary_file = comparison_dir / "experiment_summary.csv"
        
#         # Check if file exists to write header
#         write_header = not summary_file.exists()
        
#         with open(summary_file, 'a', newline='') as f:
#             writer = csv.writer(f)
            
#             if write_header:
#                 writer.writerow([
#                     'experiment_id', 'timestamp', 'model_name', 
#                     'val_mae', 'val_r2', 'val_accuracy_1%', 'val_accuracy_5%', 'val_accuracy_10%', 'val_accuracy_20%',
#                     'test_mae', 'test_r2', 'test_accuracy_1%', 'test_accuracy_5%', 'test_accuracy_10%', 'test_accuracy_20%',
#                     'total_training_time_seconds', 'average_inference_time_seconds',
#                     'total_parameters', 'trainable_parameters', 'estimated_flops',
#                     'average_memory_usage_mb', 'average_cpu_percent', 'total_energy_joules', 'best_epoch',
#                     'epochs', 'learning_rate', 'batch_size'
#                 ])
            
#             # Extract model complexity
#             complexity = results.get('model_complexity', {})
#             total_params = complexity.get('total_parameters', 0)
#             trainable_params = complexity.get('trainable_parameters', 0)
#             flops = complexity.get('estimated_flops', 0)
            
#             writer.writerow([
#                 results['experiment_id'],
#                 results['timestamp'],
#                 results['model_name'],
#                 results['mae'],
#                 results['r2'],
#                 results['accuracy_1%'],
#                 results['accuracy_5%'],
#                 results['accuracy_10%'],
#                 results['accuracy_20%'],
#                 results['test_mae'],
#                 results['test_r2'],
#                 results['test_accuracy_1%'],
#                 results['test_accuracy_5%'],
#                 results['test_accuracy_10%'],
#                 results['test_accuracy_20%'],
#                 results['total_training_time_seconds'],
#                 results['average_inference_time_seconds'],
#                 total_params,
#                 trainable_params,
#                 flops,
#                 results['average_memory_usage_mb'],
#                 results['average_cpu_percent'],
#                 results['total_energy_joules'],
#                 results['best_epoch'],
#                 results['epochs'],
#                 results['learning_rate'],
#                 results['batch_size']
#             ])
    
#     def save_results(self, results, outdir, model_name, exp_name=None):
#         """Save results to JSON file"""
#         filename = f"results_{model_name}_{self.experiment_id}"
#         if exp_name:
#             filename += f"_{exp_name}"
#         filename += ".json"
        
#         filepath = outdir / filename
        
#         # Convert numpy types to Python native types for JSON serialization
#         def convert_types(obj):
#             if isinstance(obj, (np.float32, np.float64)):
#                 return float(obj)
#             elif isinstance(obj, (np.int32, np.int64)):
#                 return int(obj)
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif isinstance(obj, list):
#                 return [convert_types(x) for x in obj]
#             elif isinstance(obj, dict):
#                 return {k: convert_types(v) for k, v in obj.items()}
#             else:
#                 return obj
        
#         results_serializable = convert_types(results)
        
#         with open(filepath, 'w') as f:
#             json.dump(results_serializable, f, indent=2)
        
#         # Also save comparison data
#         self.save_comparison_data(results, outdir, model_name, exp_name)
        
#         return filepath
    
#     def create_charts(self, results, outdir, model_name, exp_name=None):
#         """Create comprehensive charts from the collected metrics"""
#         try:
#             # Create charts directory
#             charts_dir = outdir / "charts"
#             charts_dir.mkdir(exist_ok=True)
            
#             # 1. Training and Validation Loss
#             plt.figure(figsize=(12, 4))
#             plt.subplot(1, 2, 1)
#             epochs_range = range(1, len(self.train_losses) + 1)
#             plt.plot(epochs_range, self.train_losses, label='Training Loss')
#             if self.val_losses:
#                 plt.plot(epochs_range, self.val_losses, label='Validation Loss')
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#             plt.title('Training and Validation Loss')
#             plt.legend()
#             plt.grid(True)
            
#             # 2. MAE and R²
#             plt.subplot(1, 2, 2)
#             if self.train_mae and self.val_mae:
#                 epochs_range_metrics = range(1, min(len(self.train_mae), len(self.val_mae)) + 1)
#                 plt.plot(epochs_range_metrics, self.train_mae[:len(epochs_range_metrics)], label='Train MAE')
#                 plt.plot(epochs_range_metrics, self.val_mae[:len(epochs_range_metrics)], label='Val MAE')
#                 plt.xlabel('Epoch')
#                 plt.ylabel('MAE')
#                 plt.title('Mean Absolute Error')
#                 plt.legend()
#                 plt.grid(True)
            
#             plt.tight_layout()
#             loss_chart_path = charts_dir / f"training_metrics_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
#             plt.savefig(loss_chart_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # 3. Accuracy Metrics
#             if self.val_accuracies:
#                 plt.figure(figsize=(10, 6))
#                 accuracies = np.array(self.val_accuracies)
#                 epochs_range_acc = range(1, len(accuracies) + 1)
                
#                 plt.plot(epochs_range_acc, accuracies[:, 0], label='±1% Accuracy', marker='o')
#                 plt.plot(epochs_range_acc, accuracies[:, 1], label='±5% Accuracy', marker='s')
#                 plt.plot(epochs_range_acc, accuracies[:, 2], label='±10% Accuracy', marker='^')
#                 plt.plot(epochs_range_acc, accuracies[:, 3], label='±20% Accuracy', marker='d')
                
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Accuracy')
#                 plt.title('Validation Accuracy at Different Error Bounds')
#                 plt.legend()
#                 plt.grid(True)
                
#                 accuracy_chart_path = charts_dir / f"accuracy_metrics_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
#                 plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight')
#                 plt.close()
            
#             # 4. System Metrics
#             if self.epoch_times and self.memory_usage and self.cpu_usage:
#                 plt.figure(figsize=(12, 8))
                
#                 # Epoch times
#                 plt.subplot(2, 2, 1)
#                 plt.plot(epochs_range, self.epoch_times)
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Time (s)')
#                 plt.title('Epoch Duration')
#                 plt.grid(True)
                
#                 # Memory usage
#                 plt.subplot(2, 2, 2)
#                 plt.plot(epochs_range, self.memory_usage)
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Memory (MB)')
#                 plt.title('Memory Usage')
#                 plt.grid(True)
                
#                 # CPU usage
#                 plt.subplot(2, 2, 3)
#                 plt.plot(epochs_range, self.cpu_usage)
#                 plt.xlabel('Epoch')
#                 plt.ylabel('CPU (%)')
#                 plt.title('CPU Usage')
#                 plt.grid(True)
                
#                 # Energy consumption (if available)
#                 plt.subplot(2, 2, 4)
#                 if self.energy_consumption:
#                     plt.plot(epochs_range, self.energy_consumption)
#                     plt.xlabel('Epoch')
#                     plt.ylabel('Energy (J)')
#                     plt.title('Energy Consumption')
#                 else:
#                     plt.text(0.5, 0.5, 'Energy data\nnot available', 
#                             ha='center', va='center', transform=plt.gca().transAxes)
#                 plt.grid(True)
                
#                 plt.tight_layout()
#                 system_chart_path = charts_dir / f"system_metrics_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
#                 plt.savefig(system_chart_path, dpi=300, bbox_inches='tight')
#                 plt.close()
            
#             # 5. Final Prediction vs Target Scatter Plot (using BEST epoch data)
#             if self.best_val_targets and self.best_val_predictions:
#                 plt.figure(figsize=(8, 6))
#                 plt.scatter(self.best_val_targets, self.best_val_predictions, alpha=0.6)
                
#                 # Perfect prediction line
#                 min_val = min(min(self.best_val_targets), min(self.best_val_predictions))
#                 max_val = max(max(self.best_val_targets), max(self.best_val_predictions))
#                 plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
#                 plt.xlabel('True Values')
#                 plt.ylabel('Predictions')
#                 plt.title(f'Best Epoch Predictions vs True Values\nMAE: {results["mae"]:.6f}, R²: {results["r2"]:.4f}')
#                 plt.grid(True)
                
#                 scatter_chart_path = charts_dir / f"predictions_scatter_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
#                 plt.savefig(scatter_chart_path, dpi=300, bbox_inches='tight')
#                 plt.close()
            
#             # 6. Test set predictions scatter plot
#             if self.test_targets and self.test_predictions:
#                 plt.figure(figsize=(8, 6))
#                 plt.scatter(self.test_targets, self.test_predictions, alpha=0.6)
                
#                 # Perfect prediction line
#                 min_val = min(min(self.test_targets), min(self.test_predictions))
#                 max_val = max(max(self.test_targets), max(self.test_predictions))
#                 plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
#                 plt.xlabel('True Values')
#                 plt.ylabel('Predictions')
#                 plt.title(f'Test Set Predictions vs True Values\nMAE: {results["test_mae"]:.6f}, R²: {results["test_r2"]:.4f}')
#                 plt.grid(True)
                
#                 test_scatter_path = charts_dir / f"test_predictions_scatter_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
#                 plt.savefig(test_scatter_path, dpi=300, bbox_inches='tight')
#                 plt.close()
            
#             # 7. Model Complexity Chart
#             if self.model_complexity:
#                 plt.figure(figsize=(10, 6))
#                 metrics = ['Total Parameters', 'Trainable Parameters', 'Estimated FLOPS']
#                 values = [
#                     self.model_complexity.get('total_parameters', 0),
#                     self.model_complexity.get('trainable_parameters', 0),
#                     self.model_complexity.get('estimated_flops', 0)
#                 ]
                
#                 # Normalize for better visualization
#                 values_normalized = [v / max(values) if max(values) > 0 else 0 for v in values]
                
#                 bars = plt.bar(metrics, values_normalized)
#                 plt.ylabel('Normalized Value')
#                 plt.title('Model Complexity Metrics')
#                 plt.xticks(rotation=45)
                
#                 # Add value labels on bars
#                 for bar, value in zip(bars, values):
#                     height = bar.get_height()
#                     plt.text(bar.get_x() + bar.get_width()/2., height,
#                             f'{value:,}', ha='center', va='bottom')
                
#                 plt.tight_layout()
#                 complexity_chart_path = charts_dir / f"model_complexity_{model_name}_{self.experiment_id}{f'_{exp_name}' if exp_name else ''}.png"
#                 plt.savefig(complexity_chart_path, dpi=300, bbox_inches='tight')
#                 plt.close()
            
#             print(f"Charts saved to: {charts_dir}")
#             return charts_dir
            
#         except Exception as e:
#             print(f"Warning: Could not create charts: {e}")
#             return None


# def _train(model_module, model, gs, latencies, optimizer, criterion, normalize=False, augments=None):
#     adjacency, features, latency, aug = infer.prepare_tensors(gs, latencies, model_module, model.binary_classifier, normalize, augments=augments)

#     model.train()
#     optimizer.zero_grad()
#     if augments is not None:
#         predictions = model(adjacency, features, aug)
#     else:
#         predictions = model(adjacency, features)

#     loss = criterion(predictions, latency)
#     loss.backward()
#     optimizer.step()

#     return loss


# def _test(model_module, model, g, latency, leeways, criterion, log_file=None, augments=None):
#     if not model.binary_classifier:
#         adjacency, features, latency, aug = infer.prepare_tensors([g], [latency], model_module, False, False, augments=augments)
#     else:
#         adjacency, features, latency, aug = infer.prepare_tensors(g, latency, model_module, model.binary_classifier, False, augments=augments)

#     torch.set_grad_enabled(False)
#     model.eval()
#     if augments is not None:
#         predictions = model(adjacency, features, aug)
#     else:
#         predictions = model(adjacency, features)

#     if not model.binary_classifier:
#         if log_file is not None:
#             log_file.write(f'{latency.item()} {predictions.item()} {g}\n')

#     loss = criterion(predictions, latency)
#     torch.set_grad_enabled(True)

#     if not model.binary_classifier:
#         results = []
#         for l in leeways:
#             results.append(utils.valid(predictions, latency, leeway=l))

#         return results, loss, (latency.item(), predictions.item())
#     else:
#         return None, loss, None


# def train(training_set,
#         validation_set,
#         outdir,
#         device_name,
#         model_name,
#         metric,
#         predictor_name,
#         predictor,
#         tensorboard,
#         epochs,
#         learning_rate,
#         weight_decay,
#         lr_patience,
#         es_patience,
#         batch_size,
#         shuffle,
#         optim_name,
#         lr_scheduler,
#         exp_name=None,
#         reset_last=False,
#         warmup=0,
#         save=True,
#         augments=None,
#         test_set=None):  # Added test_set parameter
    
#     model_module = importlib.import_module('.' + model_name, 'eagle.models')

#     # MODIFIED: Create the full_gcn_with_metrics folder in the project results directory
#     project_root = pathlib.Path(__file__).parent.parent.parent  # Go up to project root
#     full_gcn_dir = project_root / "results" / "full_gcn_with_metrics"
#     full_gcn_dir.mkdir(parents=True, exist_ok=True)
    
#     # Use the full_gcn_with_metrics directory for all outputs
#     outdir = full_gcn_dir / model_name / metric / device_name / predictor_name
#     outdir.mkdir(parents=True, exist_ok=True)

#     # Initialize metrics collector with experiment ID
#     experiment_id = f"{model_name}_{predictor_name}_{int(time.time())}"
#     if exp_name:
#         experiment_id = f"{exp_name}_{experiment_id}"
    
#     metrics_collector = MetricsCollector(experiment_id=experiment_id)
#     total_training_start_time = time.time()

#     # Compute model complexity
#     if training_set:
#         try:
#             # Get a sample for complexity analysis
#             sample_g, sample_latency = training_set[0]
#             sample_adjacency, sample_features, _, _ = infer.prepare_tensors(
#                 [sample_g], [sample_latency], model_module, predictor.binary_classifier, False, augments=augments
#             )
            
#             complexity_info = metrics_collector.complexity_analyzer.compute_model_complexity(
#                 predictor, (sample_adjacency, sample_features)
#             )
#             metrics_collector.set_model_complexity(complexity_info)
#             print(f"Model complexity: {complexity_info}")
#         except Exception as e:
#             print(f"Warning: Could not compute model complexity: {e}")

#     if tensorboard:
#         import torch.utils.tensorboard as tb
#         handler = tb.SummaryWriter(f'tensorboard/{exp_name}')

#     if reset_last:
#         predictor.reset_last()

#     if optim_name == 'adamw':
#         optimizer = optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     else:
#         raise ValueError(f'Unknown optimizer: {optim_name}')

#     if lr_scheduler == 'plateau':
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, threshold=0.01)
#     elif lr_scheduler == 'cosine':
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
#     else:
#         raise ValueError(f'Unknown lr scheduler: {lr_scheduler}')

#     if not predictor.binary_classifier:
#         criterion = torch.nn.L1Loss(reduction='sum')
#     else:
#         if predictor.binary_classifier == 'oneway' or predictor.binary_classifier == 'oneway-hard':
#             criterion = torch.nn.BCELoss(reduction='sum')
#         else:
#             criterion = torch.nn.KLDivLoss(reduction='sum')

#     es = utils.EarlyStopping(mode='min', patience=es_patience)

#     if predictor.binary_classifier:
#         training_set = utils.ProductList(training_set)
#         def collate_fn(batch):
#             return [[e[0] for e in pair] for pair in batch], [[e[1] for e in pair] for pair in batch]
#     else:
#         def collate_fn(batch):
#             return [e[0] for e in batch], [e[1] for e in batch]

#     training_data = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
#     data = validation_set

#     train_corrects = [0, 0, 0, 0]
#     test_corrects = [0, 0, 0, 0]
#     best_accuracies = [0, 0, 0, 0]
#     best_epochs = [0, 0, 0, 0]
#     leeways = [0.01, 0.05, 0.1, 0.2] # +-% Accuracies
#     lowest_loss = None

#     if warmup:
#         print(f'Warming up the last layer for {warmup} epochs')
#         warmup_opt = optim.AdamW(predictor.final_params(), lr=learning_rate, weight_decay=0)
#         for warmup_epoch in range(warmup):
#             print(f"Warmup Epoch: {warmup_epoch}")
#             for g, latency in training_data:
#                 loss = _train(model_module, predictor, g, latency, warmup_opt, criterion, augments=augments)

#     for epoch_no in range(epochs):
#         print(f"Epoch: {epoch_no}")
        
#         # Start epoch metrics collection
#         metrics_collector.start_epoch()

#         # Training phase
#         epoch_train_loss = 0.0
#         train_targets = []
#         train_predictions = []
        
#         for g, latency in training_data:
#             loss = _train(model_module, predictor, g, latency, optimizer, criterion, augments=augments)
#             epoch_train_loss += loss.item()

#         # Evaluate on training set
#         if not predictor.binary_classifier:
#             for g, latency in training_set:
#                 corrects, loss, (target, pred) = _test(model_module, predictor, g, latency, leeways, criterion, augments=augments)
#                 for i, c in enumerate(corrects):
#                     train_corrects[i] += c
#                 epoch_train_loss += loss.item()
#                 if target is not None and pred is not None:
#                     train_targets.append(target)
#                     train_predictions.append(pred)
#         else:
#             for g, latency in training_data:
#                 _, loss, _ = _test(model_module, predictor, g, latency, None, criterion, augments=augments)
#                 epoch_train_loss += loss.item()

#         avg_train_loss = epoch_train_loss / len(training_set) if training_set else 0
        
#         # Record training metrics
#         if train_targets and train_predictions:
#             metrics_collector.record_train_metrics(avg_train_loss, train_targets, train_predictions)

#         # Validation phase
#         epoch_val_loss = 0.0
#         val_targets = []
#         val_predictions = []
        
#         if not predictor.binary_classifier:
#             # Measure inference time for validation
#             inference_start_time = time.time()
            
#             for g, latency in validation_set:
#                 corrects, loss, (target, pred) = _test(model_module, predictor, g, latency, leeways, criterion, augments=augments)
#                 for i, c in enumerate(corrects):
#                     test_corrects[i] += c
#                 epoch_val_loss += loss.item()
#                 if target is not None and pred is not None:
#                     val_targets.append(target)
#                     val_predictions.append(pred)
            
#             inference_time = time.time() - inference_start_time
#             metrics_collector.record_inference_time(inference_time)
            
#             avg_val_loss = epoch_val_loss / len(validation_set) if validation_set else 0

#             if validation_set:
#                 current_accuracies = [test_correct / len(validation_set) for test_correct in test_corrects]
#             else:
#                 current_accuracies = [0, 0, 0, 0]
#             print(f'Average loss of validation set {epoch_no}: {avg_val_loss}')
            
#             # **FIXED: Pass epoch number to track best epoch**
#             metrics_collector.record_val_metrics(avg_val_loss, val_targets, val_predictions, epoch_no)

#             for i, best_accuracy in enumerate(best_accuracies):
#                 if current_accuracies[i] >= best_accuracy:
#                     best_accuracies[i] = current_accuracies[i]
#                     best_epochs[i] = epoch_no
#         else:
#             avg_val_loss = epoch_train_loss  # For binary classifiers, use train loss as val loss

#         # End epoch metrics collection
#         metrics_collector.end_epoch()

#         # Print metrics
#         if not predictor.binary_classifier and training_set:
#             train_accuracies = [train_correct / len(training_set) for train_correct in train_corrects]
#             print(f'Top +-{leeways} Accuracy of train set for epoch {epoch_no}: {train_accuracies} ')
#             if validation_set:
#                 print(f'Top +-{leeways} Accuracy of validation set for epoch {epoch_no}: {current_accuracies}')
#                 print(f'[best: {best_accuracies} @ epoch {best_epochs}]')
            
#             # Print additional metrics
#             if (metrics_collector.train_mae and epoch_no < len(metrics_collector.train_mae) and 
#                 metrics_collector.val_mae and epoch_no < len(metrics_collector.val_mae)):
#                 print(f'Train MAE: {metrics_collector.train_mae[-1]:.6f}, Val MAE: {metrics_collector.val_mae[-1]:.6f}')
#                 print(f'Train R²: {metrics_collector.train_r2[-1]:.4f}, Val R²: {metrics_collector.val_r2[-1]:.4f}')
        
#         print(f'Average loss of training set {epoch_no}: {avg_train_loss:.6f}')
#         if metrics_collector.epoch_times:
#             print(f'Epoch time: {metrics_collector.epoch_times[-1]:.2f}s')
#         if PSUTIL_AVAILABLE and metrics_collector.memory_usage:
#             print(f'Memory usage: {metrics_collector.memory_usage[-1]:.2f} MB')
#             print(f'CPU usage: {metrics_collector.cpu_usage[-1]:.2f}%')
#         if PYRAPL_AVAILABLE and metrics_collector.energy_consumption:
#             print(f'Energy consumption: {metrics_collector.energy_consumption[-1]:.2f} J')
#         if metrics_collector.inference_times:
#             print(f'Inference time: {metrics_collector.inference_times[-1]:.4f}s')

#         # Model checkpointing
#         if lowest_loss is None or avg_val_loss < lowest_loss:
#             lowest_loss = avg_val_loss
#             best_predictor_weight = predictor.state_dict()
#             if save:
#                 torch.save(best_predictor_weight, outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt'))
#             print(f'Lowest val_loss: {avg_val_loss:.6f}... Predictor model saved.')

#         # Learning rate scheduling
#         if lr_scheduler == 'plateau':
#             if epoch_no > 20:
#                 scheduler.step(avg_val_loss)
#         else:
#             scheduler.step()

#         # Early stopping
#         if epoch_no > 20:
#             if es.step(avg_val_loss):
#                 print('Early stopping criterion is met, stop training now.')
#                 break

#         # Reset counters for next epoch
#         train_corrects = [0, 0, 0, 0]
#         test_corrects = [0, 0, 0, 0]

#         # Tensorboard logging
#         if tensorboard:
#             handler.add_scalar('loss/training', avg_train_loss, epoch_no)
#             handler.add_scalar('loss/validation', avg_val_loss, epoch_no)
#             if not predictor.binary_classifier and training_set and validation_set:
#                 handler.add_scalar('accuracy_1/training', train_accuracies[0], epoch_no)
#                 handler.add_scalar('accuracy_1/validation', current_accuracies[0], epoch_no)
#                 handler.add_scalar('accuracy_5/training', train_accuracies[1], epoch_no)
#                 handler.add_scalar('accuracy_5/validation', current_accuracies[1], epoch_no)
#                 handler.add_scalar('accuracy_10/training', train_accuracies[2], epoch_no)
#                 handler.add_scalar('accuracy_10/validation', current_accuracies[2], epoch_no)
#                 handler.add_scalar('accuracy_20/training', train_accuracies[3], epoch_no)
#                 handler.add_scalar('accuracy_20/validation', current_accuracies[3], epoch_no)
#                 if (metrics_collector.train_mae and epoch_no < len(metrics_collector.train_mae) and
#                     metrics_collector.val_mae and epoch_no < len(metrics_collector.val_mae)):
#                     handler.add_scalar('metrics/mae_train', metrics_collector.train_mae[-1], epoch_no)
#                     handler.add_scalar('metrics/mae_val', metrics_collector.val_mae[-1], epoch_no)
#                     handler.add_scalar('metrics/r2_train', metrics_collector.train_r2[-1], epoch_no)
#                     handler.add_scalar('metrics/r2_val', metrics_collector.val_r2[-1], epoch_no)
#             if metrics_collector.epoch_times:
#                 handler.add_scalar('system/epoch_time', metrics_collector.epoch_times[-1], epoch_no)
#             if PSUTIL_AVAILABLE and metrics_collector.memory_usage:
#                 handler.add_scalar('system/memory_usage', metrics_collector.memory_usage[-1], epoch_no)
#                 handler.add_scalar('system/cpu_usage', metrics_collector.cpu_usage[-1], epoch_no)
#             if PYRAPL_AVAILABLE and metrics_collector.energy_consumption:
#                 handler.add_scalar('system/energy_consumption', metrics_collector.energy_consumption[-1], epoch_no)
#             if metrics_collector.inference_times:
#                 handler.add_scalar('system/inference_time', metrics_collector.inference_times[-1], epoch_no)

#     # End of training
#     total_training_time = time.time() - total_training_start_time
    
#     if tensorboard:
#         handler.close()
    
#     if save:
#         torch.save(best_predictor_weight, outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt'))

#     # Evaluate on test set if provided
#     if test_set is not None and not predictor.binary_classifier:
#         print("\n=== FINAL TEST SET EVALUATION ===")
#         test_targets = []
#         test_predictions = []
#         test_loss = 0.0
        
#         inference_start_time = time.time()
#         for g, latency in test_set:
#             _, loss, (target, pred) = _test(model_module, predictor, g, latency, leeways, criterion, augments=augments)
#             test_loss += loss.item()
#             if target is not None and pred is not None:
#                 test_targets.append(target)
#                 test_predictions.append(pred)
        
#         inference_time = time.time() - inference_start_time
#         metrics_collector.record_inference_time(inference_time)
        
#         if test_targets and test_predictions:
#             metrics_collector.record_test_metrics(test_targets, test_predictions)
#             print(f"Test set evaluation completed:")
#             print(f"Test MAE: {metrics_collector.test_mae:.6f}")
#             print(f"Test R²: {metrics_collector.test_r2:.4f}")
#             print(f"Test Accuracy 1%: {metrics_collector.test_accuracies[0]:.4f}")
#             print(f"Test Accuracy 5%: {metrics_collector.test_accuracies[1]:.4f}")
#             print(f"Test Accuracy 10%: {metrics_collector.test_accuracies[2]:.4f}")
#             print(f"Test Accuracy 20%: {metrics_collector.test_accuracies[3]:.4f}")

#     # Save final results
#     cfg = {
#         'epochs': epochs,
#         'learning_rate': learning_rate,
#         'weight_decay': weight_decay,
#         'batch_size': batch_size
#     }
    
#     results = metrics_collector.get_final_results(cfg, model_name, total_training_time)
#     results_filepath = metrics_collector.save_results(results, outdir, model_name, exp_name)
    
#     # Create charts
#     charts_dir = metrics_collector.create_charts(results, outdir, model_name, exp_name)
    
#     print("Training finished!")
#     print(f"Results saved to: {results_filepath}")
#     if charts_dir:
#         print(f"Charts saved to: {charts_dir}")
    
#     # Print summary of final results
#     print("\n=== TRAINING SUMMARY ===")
#     print(f"Experiment ID: {experiment_id}")
#     print(f"Total training time: {total_training_time:.2f}s")
#     print(f"Best validation epoch: {results['best_epoch']}")
#     print(f"Best validation loss: {results['best_val_loss']:.6f}")
#     print(f"Final validation MAE: {results['mae']:.6f}")
#     print(f"Final validation R²: {results['r2']:.4f}")
#     print(f"Validation Accuracy 1%: {results['accuracy_1%']:.4f}")
#     print(f"Validation Accuracy 5%: {results['accuracy_5%']:.4f}")
#     print(f"Validation Accuracy 10%: {results['accuracy_10%']:.4f}")
#     print(f"Validation Accuracy 20%: {results['accuracy_20%']:.4f}")
    
#     if test_set is not None:
#         print(f"Test MAE: {results['test_mae']:.6f}")
#         print(f"Test R²: {results['test_r2']:.4f}")
#         print(f"Test Accuracy 1%: {results['test_accuracy_1%']:.4f}")
#         print(f"Test Accuracy 5%: {results['test_accuracy_5%']:.4f}")
#         print(f"Test Accuracy 10%: {results['test_accuracy_10%']:.4f}")
#         print(f"Test Accuracy 20%: {results['test_accuracy_20%']:.4f}")
    
#     # Model complexity summary
#     if metrics_collector.model_complexity:
#         print(f"Total parameters: {metrics_collector.model_complexity.get('total_parameters', 0):,}")
#         print(f"Trainable parameters: {metrics_collector.model_complexity.get('trainable_parameters', 0):,}")
#         print(f"Estimated FLOPS: {metrics_collector.model_complexity.get('estimated_flops', 0):,}")
    
#     # System metrics summary
#     if PSUTIL_AVAILABLE:
#         print(f"Average memory usage: {results['average_memory_usage_mb']:.2f} MB")
#         print(f"Average CPU usage: {results['average_cpu_percent']:.2f}%")
#     if PYRAPL_AVAILABLE:
#         print(f"Total energy consumption: {results['total_energy_joules']:.2f} J")
#         print(f"Average energy per epoch: {results['average_energy_per_epoch_joules']:.2f} J")
    
#     print(f"Average inference time: {results['average_inference_time_seconds']:.4f}s")

#     predictor.load_state_dict(best_predictor_weight)
#     return predictor, results_filepath


# def predict(testing_data,
#         outdir,
#         device_name,
#         model_name,
#         metric,
#         predictor_name,
#         predictor,
#         log=False,
#         exp_name=None,
#         load=False,
#         iteration=None,
#         explored_models=None,
#         valid_pts=None,
#         use_fast=True,
#         augments=None):
#     model_module = importlib.import_module('.' + model_name, 'eagle.models')

#     # MODIFIED: Create the full_gcn_with_metrics folder in the project results directory
#     project_root = pathlib.Path(__file__).parent.parent.parent  # Go up to project root
#     full_gcn_dir = project_root / "results" / "full_gcn_with_metrics"
#     full_gcn_dir.mkdir(parents=True, exist_ok=True)
    
#     # Use the full_gcn_with_metrics directory for all outputs
#     if load or log:
#         outdir = full_gcn_dir / model_name / metric / device_name / predictor_name
#         if log:
#             outdir.mkdir(parents=True, exist_ok=True)

#     if load and predictor_name != 'random':
#         predictor.load_state_dict(torch.load(outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt')))
#         print('Predictor imported.')

#     criterion = torch.nn.L1Loss()

#     test_corrects = [0, 0, 0, 0]
#     leeways = [0.01, 0.05, 0.1, 0.2]

#     log_file = None
#     if log:
#         log_filename = 'log.txt' if exp_name is None else f'log_{exp_name}.txt'
#         if iteration is not None:
#             log_filename = f'iter{iteration}_' + log_filename

#         log_file = outdir / log_filename
#         sep = False
#         if log_file.exists():
#             sep = True
#         log_file = log_file.open('a')
#         if sep:
#             log_file.write('===\n')

#     if predictor_name == 'random':
#         print('Producing random ordering of the dataset...')
#         predicted = []
#         perm = np.random.permutation(len(testing_data))
#         for idx, (point, gt_value) in enumerate(testing_data):
#             predicted_value = perm[idx]
#             predicted.append(predicted_value)
#             if log:
#                 log_file.write(f'{gt_value} {predicted_value} {point}\n')

#     elif not predictor.binary_classifier:
#         predicted = []
#         test_loss = 0
#         test_targets = []
#         test_predictions = []
        
#         for g, latency in testing_data:
#             corrects, loss, values = _test(model_module, predictor, g, latency, leeways, criterion, log_file, augments)
#             for i, c in enumerate(corrects):
#                 test_corrects[i] += c

#             test_loss += loss
#             predicted.append(values[1])
#             test_targets.append(values[0])
#             test_predictions.append(values[1])

#         current_accuracies = [test_correct / len(testing_data) for test_correct in test_corrects]
#         avg_loss = test_loss / len(testing_data)

#         print(f'Top +-{leeways} Accuracy of test set: {current_accuracies}')
#         print(f'Average loss of test set: {avg_loss}')
        
#         # Calculate and print additional test metrics
#         if test_targets and test_predictions:
#             from sklearn.metrics import mean_absolute_error, r2_score
#             test_mae = mean_absolute_error(test_targets, test_predictions)
#             test_r2 = r2_score(test_targets, test_predictions)
#             print(f'Test MAE: {test_mae:.6f}')
#             print(f'Test R²: {test_r2:.4f}')
#     else:
#         torch.set_grad_enabled(False)
#         predictor.eval()

#         if use_fast:
#             print(f'Precomputing embeddings for {len(testing_data)} graphs')
#             precomputed = infer.precompute_embeddings(model_module, predictor, testing_data, 1024, augments=augments)
#             print('Done')

#         total = 0
#         correct = 0
#         skipped = 0
#         def predictor_compare(v1, v2):
#             nonlocal total
#             nonlocal correct
#             nonlocal skipped
#             total += 1
#             if valid_pts is not None and v1[0] not in valid_pts:
#                 skipped += 1
#                 return -1
#             if valid_pts is not None and v2[0] not in valid_pts:
#                 skipped += 1
#                 return 1
#             latencies = [v1[1], v2[1]]
#             if use_fast:
#                 result = infer.precomputed_forward(predictor, [v1[2], v2[2]], precomputed)
#             else:
#                 gs = [v1[0], v2[0]]
#                 adjacency, features, _, aug = infer.prepare_tensors([gs], None, model_module, predictor.binary_classifier, False, augments=augments)
#                 if augments is not None:
#                     result = predictor(adjacency, features, aug)
#                 else:
#                     result = predictor(adjacency, features)
#             if predictor.binary_classifier == 'oneway' or predictor.binary_classifier == 'oneway-hard':
#                 v1_better = result[0][0].cpu().item() - 0.5
#                 if latencies[0] > latencies[1]:
#                     if v1_better > 0:
#                         correct += 1
#                 elif v1_better < 0:
#                     correct += 1

#                 return v1_better
#             else:
#                 rv1, rv2 = result[0][0].cpu().item(), result[0][1].cpu().item()
#                 if latencies[0] > latencies[1]:
#                     if rv1 > rv2:
#                         correct += 1
#                 elif rv1 < rv2:
#                     correct += 1

#                 # we want higher number to appear later (have higher "score"), so (v1 - v2) should get us the correct order
#                 return rv1 - rv2

#         if use_fast:
#             predictor.cpu()
#             test_data_with_indices = [(*v, idx) for idx, v in enumerate(testing_data)]
#             sorted_values = sorted(test_data_with_indices, key=functools.cmp_to_key(predictor_compare))
#             sorted_values = { pt: (gt,idx) for idx,(pt,gt,_) in enumerate(sorted_values) }
#             # predictor.cuda()
#         else:
#             sorted_values = sorted(testing_data, key=functools.cmp_to_key(predictor_compare))
#             sorted_values = { pt: (gt,idx) for idx,(pt,gt) in enumerate(sorted_values) }

#         predicted = []
#         for p, v in testing_data:
#             r = sorted_values[p][1]
#             predicted.append(r)
#             if log:
#                 log_file.write(f'{v} {r} {p}\n')

#         predictor.train()
#         torch.set_grad_enabled(True)

#     if log:
#         log_file.write('---\n')
#         explored_models = explored_models or []
#         for p,v in explored_models:
#             log_file.write(f'{p}\n')
#         log_file.write('---\n')
#         if predictor_name == 'random':
#             pass
#         elif not predictor.binary_classifier:
#             log_file.write(f'{avg_loss}\n{current_accuracies}\n')
#         else:
#             log_file.write(f'{correct}/{total} predictions correct\n')
#             log_file.write(f'{skipped}/{total} predictions skipped\n')
#         log_file.close()

#     return predicted

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, required=True, help='Model family to run, should be a name of one of the packages under eagle.models')
#     parser.add_argument('--device', type=str, required=True, help='Device on which the models will be run, should be a name of one of the packages under eagle.device_runner')
#     parser.add_argument('--metric', type=str, default='latency', help='Metric to measure. Default: latency.')
#     parser.add_argument('--predictor', type=str, required=True, help='Predictor to train, should a name of one of the packages under eagle.predictors')
#     parser.add_argument('--measurement', type=str, required=True, default=None, help='Measurement file for device')
#     parser.add_argument('--cfg', type=str, default=None, help='Configuration file for device and model packages')
#     parser.add_argument('--expdir', type=str, default='results', help='Folder in which the results of measurements will be saved. Default: results')
#     parser.add_argument('--process', action='store_true', help='Process measurements - use this if the measurements are not already processed')
#     parser.add_argument('--multiple_files', action='store_true', help='Combine results from multiple files - use this if the measurements are not already combined')
#     parser.add_argument('--transfer', default=None, help='Perform transfer learning from a previously trained model - the argument should point to the checkpoint to load')
#     parser.add_argument('--load', default=None, help='Checkpoint to load')
#     parser.add_argument('--warmup', default=0, type=int, help='Number of warmup epochs for the last layer')
#     parser.add_argument('--foresight_warmup', type=str, help='Path to the dataset containing foresight metrics which will be used to warmup the predictor during iterative training')
#     parser.add_argument('--foresight_simple', action='store_true', help='Do not train the predictor when doing foresight warmup, instead simply rank models with foresight scores directly')
#     parser.add_argument('--foresight_augment', type=str, nargs='+', default=[], help='Path to foresight metrics, if set they will be passed to the predictor together with each model')
#     parser.add_argument('--prediction_only', action='store_true', help='Run prediction with a pretrained predictor')
#     parser.add_argument('--exp', default=None, help='Optional experiment name, used when saving the predictor to distinguish between different configurations')
#     parser.add_argument('--uid', default=None, type=int, help='UID to distinguish between different concurrent runs')
#     parser.add_argument('--log', action='store_true', help='Log prediction on test dataset together with ground truth')
#     parser.add_argument('--tensorboard', action='store_true', help='Log training data for visualization in tensorboard')
#     parser.add_argument('--torch_seed', type=int, default=None, help='Fixed seed to use with torch.random')
#     parser.add_argument('--quiet', action='store_true', help='Suppress standard output')
#     parser.add_argument('--iter', type=int, default=0, help='Number of iterations when using iterative search')
#     parser.add_argument('--save', action='store_true', help='Save the best predictor')
#     parser.add_argument('--eval', action='store_true', help='Eval model only, do not train (use with --transfer to eval pretrained model)')
#     parser.add_argument('--lat_limit', type=float, default=None, help='Latency limit to prune the search space (requires --transfer to point to the latency predictor)')
#     parser.add_argument('--sample_best', action='store_true')
#     parser.add_argument('--sample_best2', action='store_true')
#     parser.add_argument('--reset_last', action='store_true', help='Reset last layer (only applicable if checkpoint is loaded)')

#     parser.add_argument('--leave_one_out', type=str)
#     parser.add_argument('--dataset_path', type=str)
#     args = parser.parse_args()

#     if args.uid is not None:
#         if args.exp is None:
#             args.exp = str(args.uid)
#         else:
#             args.exp += f'_{args.uid}'

#     with contextlib.ExitStack() as es:
#         if args.quiet:
#             f = es.enter_context(open(os.devnull, 'w'))
#             es.enter_context(contextlib.redirect_stdout(f))

#         extra_args = {}
#         if args.cfg:
#             import yaml
#             with open(args.cfg, 'r') as f:
#                 extra_args = yaml.load(f, Loader=yaml.Loader)

#         if args.transfer:
#             if not args.load:
#                 raise ValueError('Both --load and --transfer are set, please use only one. Note: "--transfer X" is the same as "--load X --reset_last"')

#             args.load = args.transfer
#             args.reset_last = True

#         if args.predictor == 'random':
#             predictor = None
#         else:
#             predictor = infer.get_predictor(args.predictor, predictor_args=extra_args.get('predictor'), checkpoint=args.load, ignore_last=args.reset_last, augment=len(args.foresight_augment))
#         lat_predictor = None
#         if args.lat_limit:
#             if not args.transfer:
#                 raise ValueError('--lat_limit requires --transfer')

#             lat_predictor_args = extra_args.get('predictor').copy()
#             lat_predictor_args.pop('binary_classifier')
#             lat_predictor = infer.get_predictor(args.predictor, predictor_args=lat_predictor_args, checkpoint=args.load, ignore_last=False)

#         # if args.predictor != 'random':
#         #     if torch.cuda.is_available():
#         #         predictor.cuda()
#         #         if lat_predictor:
#         #             lat_predictor.cuda()
#             # else:
#                 # raise RuntimeError('No GPU!')

#         if args.model == 'darts':
#             dataset_args = extra_args.get('dataset', {})
#             dataset_file = dataset_args.pop('dataset_file', None)
#             if dataset_file:
#                 dataset_file = pathlib.Path(args.expdir) / args.model / args.metric / args.device / dataset_file
#             dataset = dataset_mod.DartsDataset(args.measurement,
#                                     dataset_file=dataset_file,
#                                     **extra_args.get('dataset', {}))
#         else:
#             dataset = dataset_mod.EagleDataset(args.measurement,
#                                     args.process,
#                                     args.multiple_files,
#                                     **extra_args.get('dataset', {}),
#                                     lat_limit=args.lat_limit,
#                                     lat_predictor=lat_predictor,
#                                     model_module=importlib.import_module('.' + args.model, 'eagle.models'))

#             if args.foresight_warmup:
#                 if not args.iter:
#                     raise ValueError('Foresight warmup requires iterative training!')
#                 if args.foresight_augment:
#                     raise ValueError('Foresigh augment is incompatible with foresight warmup')

#                 foresight_dataset = dataset_mod.EagleDataset(args.foresight_warmup,
#                     args.process,
#                     args.multiple_files,
#                     **extra_args.get('foresight', {}).get('dataset', {}),
#                     lat_limit=args.lat_limit,
#                     lat_predictor=lat_predictor,
#                     model_module=importlib.import_module('.' + args.model, 'eagle.models'))

#             if args.foresight_augment:
#                 print(f'Using {len(args.foresight_augment)} foresight metric(s) to augment graph embeddings')
#                 augments = []
#                 for aug in args.foresight_augment:
#                     with open(aug, 'rb') as f:
#                         d = pickle.load(f)
#                         augments.append(d)
#             else:
#                 augments = None

#         explored_models = dataset.train_set
#         if not args.eval:
#             if args.iter:
#                 if args.foresight_warmup:
#                     if not args.foresight_simple:
#                         print(f'Warming up predictor using foresight dataset {args.foresight_warmup!r}')
#                         foresight_train_args = extra_args.get('foresight', {}).get('training', {})
#                         train(foresight_dataset.train_set,
#                             foresight_dataset.valid_set,
#                             args.expdir,
#                             args.device,
#                             args.model,
#                             args.metric,
#                             args.predictor,
#                             predictor,
#                             args.tensorboard,
#                             **foresight_train_args,
#                             exp_name=args.exp,
#                             reset_last=args.reset_last,
#                             warmup=args.warmup,
#                             save=False,
#                             augments=augments)
#                     else:
#                         print(f'Sorting models using foresight metrics from: {args.foresight_warmup!r}')

#                 train_args = extra_args.get('training', {})

#                 target_batch = train_args.pop('batch_size')
#                 batch_per_iter = target_batch // args.iter
#                 current_batch = batch_per_iter

#                 target_epochs = train_args.pop('epochs')
#                 epochs_per_iter = target_epochs // args.iter
#                 current_epochs = epochs_per_iter

#                 points_per_iter = len(dataset.train_set) // args.iter
#                 candidates = list(dataset.dataset)

#                 if not args.foresight_warmup:
#                     train_set = dataset_mod.select_random(candidates, points_per_iter)
#                 else:
#                     train_set = []

#                 for i in range(args.iter):
#                     print('Iteration', i)

#                     if i or args.foresight_warmup:
#                         # update training set
#                         if i or not args.foresight_simple:
#                             scores = predict(candidates, args.expdir, args.device, args.model, args.metric, args.predictor, predictor, log=False, exp_name=args.exp, load=False, augments=augments)
#                         else:
#                             scores = [p[1] for p in foresight_dataset.dataset]
#                         if args.sample_best or args.sample_best2:
#                             if not args.sample_best2:
#                                 median_score = statistics.median(scores)
#                                 candidates = [pt for pt, score in zip(candidates, scores) if score > median_score]
#                             best_candidates = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
#                             added = 0
#                             for candidate, score in best_candidates:
#                                 if added == points_per_iter//2:
#                                     break
#                                 if candidate in train_set:
#                                     continue
#                                 train_set.append(candidate)
#                                 added += 1

#                             if args.sample_best2:
#                                 random_th = best_candidates[len(scores) // (2**(i or 1))][1]
#                                 random_candidates = [pt for pt, score in zip(candidates, scores) if score > random_th]
#                                 selected_candidates = dataset_mod.select_random(random_candidates, points_per_iter//2, current=train_set)
#                             else:
#                                 selected_candidates = dataset_mod.select_random(candidates, points_per_iter//2, current=train_set)
#                             train_set.extend(selected_candidates)
#                         else:
#                             median_score = statistics.median(scores)
#                             candidates = [pt for pt, score in zip(candidates, scores) if score > median_score]
#                             sampled = dataset_mod.select_random(candidates, points_per_iter, current=train_set)
#                             train_set.extend(sampled)

#                     print('Number of candidate points:', len(candidates))
#                     print('Number of training points:', len(train_set))
#                     print('Batch size:', current_batch)
#                     print('Number of epochs:', current_epochs)

#                     train(train_set,
#                         train_set,
#                         args.expdir,
#                         args.device,
#                         args.model,
#                         args.metric,
#                         args.predictor,
#                         predictor,
#                         args.tensorboard,
#                         **train_args,
#                         batch_size=current_batch,
#                         epochs=current_epochs,
#                         exp_name=args.exp,
#                         reset_last=args.reset_last and not i and not args.foresight_warmup,
#                         warmup=args.warmup if (not i and not args.foresight_warmup) else 0,
#                         save=args.save and i + 1 == args.iter,
#                         augments=augments)

#                     current_batch += batch_per_iter
#                     current_epochs += epochs_per_iter
#                     explored_models = train_set
#             else:
#                 if not dataset.train_set:
#                     raise ValueError('Training set is empty!')
#                 train(dataset.train_set,
#                     dataset.valid_set,
#                     args.expdir,
#                     args.device,
#                     args.model,
#                     args.metric,
#                     args.predictor,
#                     predictor,
#                     args.tensorboard,
#                     **extra_args.get('training', {}),
#                     exp_name=args.exp,
#                     reset_last=args.reset_last,
#                     warmup=args.warmup,
#                     save=args.save,
#                     augments=augments,
#                     test_set=dataset.full_dataset)  # Pass test set for final evaluation

#         predict(dataset.full_dataset,
#             args.expdir,
#             args.device,
#             args.model,
#             args.metric,
#             args.predictor,
#             predictor,
#             args.log,
#             exp_name=args.exp,
#             load=False,
#             explored_models=explored_models,
#             valid_pts=dataset.valid_pts,
#             augments=augments)




