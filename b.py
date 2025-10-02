# 2_extract_embeddings_real.py
import sys
import os
sys.path.append('.')

def extract_embeddings():
    """Extract embeddings using trained GCN"""
    try:
        print("=== EXTRACTING GCN EMBEDDINGS ===")
        
        # Load dataset
        import pickle
        with open("results/desktop-cpu-core-i7-7820x.pickle", 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"Total dataset: {len(dataset)} samples")
        
        # Load trained GCN
        import torch
        from eagle.predictors.gcn.gcn import GCN
        
        model = GCN(num_features=6, num_layers=4, num_hidden=600)
        model.load_state_dict(torch.load("results/nasbench201/latency/cpu/gcn/predictor_simple.pt"))
        
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        
        # Use same 1520 points as paper
        data_pairs = list(dataset.items())[:1520]
        train_data = data_pairs[:900]    # 900 training
        test_data = data_pairs[900:1520] # 620 testing
        
        print(f"Data for embeddings:")
        print(f"  Training: {len(train_data)}")
        print(f"  Testing: {len(test_data)}")
        
        # Extract embeddings function
        from eagle.models import nasbench201
        from eagle.predictors import infer
        import numpy as np
        
        def extract_embeddings(data, gcn_model):
            embeddings = []
            targets = []
            
            for i, (graph, target) in enumerate(data):
                if i % 100 == 0:
                    print(f"Processing: {i}/{len(data)}")
                
                try:
                    adjacency, features, _, _ = infer.prepare_tensors(
                        [graph], [target], nasbench201, False, False
                    )
                    
                    with torch.no_grad():
                        x = gcn_model.forward_single_model(adjacency, features)
                        emb = x[:, 0]
                        embeddings.append(emb.cpu().numpy().flatten())
                        targets.append(target)
                        
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            return np.array(embeddings), np.array(targets)
        
        print("Extracting training embeddings...")
        X_train, y_train = extract_embeddings(train_data, model)
        
        print("Extracting test embeddings...")
        X_test, y_test = extract_embeddings(test_data, model)
        
        print(f"Embeddings extracted:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        
        # Save embeddings
        np.save("results/gcn_embeddings_train.npy", X_train)
        np.save("results/gcn_embeddings_test.npy", X_test)
        np.save("results/gcn_targets_train.npy", y_train)
        np.save("results/gcn_targets_test.npy", y_test)
        
        print("Embeddings saved successfully!")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_embeddings()