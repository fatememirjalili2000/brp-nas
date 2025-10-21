# high_accuracy_light_gcn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from .. import utils

class AttentionGraphConvolution(Module):
    """
    Graph convolution with attention mechanism for important nodes
    """
    def __init__(self, in_features, out_features, bias=True, 
                 weight_init='thomas', bias_init='thomas'):
        super(AttentionGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Main weight matrix
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        
        # Attention weights
        self.attention_weight = Parameter(torch.DoubleTensor(out_features, 1))
        
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        utils.init_tensor(self.weight, self.weight_init, 'relu')
        utils.init_tensor(self.attention_weight, self.weight_init, 'relu')
        if self.bias is not None:
            utils.init_tensor(self.bias, self.bias_init, 'relu')

    def forward(self, adjacency, features):
        # Standard graph convolution
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        # Apply attention to highlight important nodes
        attention_scores = torch.matmul(output, self.attention_weight)
        attention_weights = F.softmax(attention_scores, dim=1)
        output = output * attention_weights
        
        return output

class HighAccuracyLightGCN(Module):
    """
    High-accuracy lightweight GCN that prioritizes accuracy over extreme efficiency
    """
    def __init__(self, 
                num_features=0, 
                num_layers=2,           # Keep shallow for efficiency
                num_hidden=48,          # WIDER for better representation (increased from 16)
                dropout_ratio=0.2,      # Moderate regularization
                weight_init='thomas',
                bias_init='thomas',
                binary_classifier=False,
                augments=0,
                use_residual=True,      # Residual connections for better gradients
                use_attention=True,     # Attention for important nodes
                activation='leaky_relu'): # Better activation

        super(HighAccuracyLightGCN, self).__init__()
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio
        self.binary_classifier = binary_classifier
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Choose convolution type
        conv_layer = AttentionGraphConvolution if use_attention else GraphConvolution
        
        # Graph convolution layers with residual connections
        self.gc = nn.ModuleList([
            conv_layer(
                self.nfeat if i==0 else self.nhid, 
                self.nhid, 
                bias=True, 
                weight_init=weight_init, 
                bias_init=bias_init
            ) for i in range(self.nlayer)
        ])
        
        # Advanced activation - LeakyReLU prevents dead neurons
        if activation == 'leaky_relu':
            self.activation = nn.ModuleList([nn.LeakyReLU(0.01).double() for _ in range(self.nlayer)])
        elif activation == 'elu':
            self.activation = nn.ModuleList([nn.ELU().double() for _ in range(self.nlayer)])
        else:
            self.activation = nn.ModuleList([nn.ReLU().double() for _ in range(self.nlayer)])
        
        # Layer normalization for stable training
        self.ln = nn.ModuleList([nn.LayerNorm(self.nhid).double() for _ in range(self.nlayer)])
        
        # Dropout for regularization
        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio).double() for _ in range(self.nlayer)])

        # Enhanced final layer with more capacity
        if not binary_classifier:
            self.fc = nn.Sequential(
                nn.Linear(self.nhid + augments, 64).double(),
                nn.LeakyReLU(0.01).double(),
                nn.Dropout(0.1).double(),
                nn.Linear(64, 32).double(),
                nn.LeakyReLU(0.01).double(),
                nn.Linear(32, 1).double()
            )
        else:
            # Keep binary classifier structure same as original
            if binary_classifier == 'naive':
                self.fc = nn.Linear(self.nhid + augments, 1).double()
            elif binary_classifier == 'oneway' or binary_classifier == 'oneway-hard':
                self.fc = nn.Linear((self.nhid + augments) * 2, 1).double()
            else:
                self.fc = nn.Linear((self.nhid + augments) * 2, 2).double()

            if binary_classifier != 'oneway' and binary_classifier != 'oneway-hard':
                self.final_act = nn.LogSoftmax(dim=1)
            else:
                self.final_act = nn.Sigmoid()

    def forward_single_model(self, adjacency, features):
        """Forward pass with residual connections"""
        x = self.gc[0](adjacency, features)
        x = self.ln[0](x)
        x = self.activation[0](x)
        x = self.dropout[0](x)
        
        for i in range(1, self.nlayer):
            residual = x  # Store for residual connection
            
            x = self.gc[i](adjacency, x)
            x = self.ln[i](x)
            x = self.activation[i](x)
            x = self.dropout[i](x)
            
            # Residual connection
            if self.use_residual:
                x = x + residual

        return x

    def extract_features(self, adjacency, features, augments=None):
        """Extract embeddings - maintains same interface"""
        x = self.forward_single_model(adjacency, features)
        x = x[:, 0]  # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
        return x

    def regress(self, features, features2=None):
        """Regression head"""
        if not self.binary_classifier:
            assert features2 is None
            return self.fc(features)

        assert features2 is not None
        if self.binary_classifier == 'naive':
            x1 = self.fc(features)
            x2 = self.fc(features2)
        else:
            x1 = features
            x2 = features2

        x = torch.cat([x1, x2], dim=1)
        if self.binary_classifier != 'naive':
            x = self.fc(x)

        x = self.final_act(x)
        return x

    def forward(self, adjacency, features, augments=None):
        """Main forward pass"""
        if not self.binary_classifier:
            x = self.forward_single_model(adjacency, features)
            x = x[:, 0]  # use global node
            if augments is not None:
                x = torch.cat([x, augments], dim=1)
            return self.fc(x)
        else:
            x1 = self.forward_single_model(adjacency[:, 0], features[:, 0])
            x1 = x1[:, 0]
            x2 = self.forward_single_model(adjacency[:, 1], features[:, 1])
            x2 = x2[:, 0]
            
            if augments is not None:
                a1 = augments[:, 0]
                a2 = augments[:, 1]
                x1 = torch.cat([x1, a1], dim=1)
                x2 = torch.cat([x2, a2], dim=1)

            if self.binary_classifier == 'naive':
                x1 = self.fc(x1)
                x2 = self.fc(x2)

            x = torch.cat([x1, x2], dim=1)
            if self.binary_classifier != 'naive':
                x = self.fc(x)

            x = self.final_act(x)
            return x

    def reset_last(self):
        """Reset final layers"""
        for layer in self.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def final_params(self):
        """Get final layer parameters"""
        return self.fc.parameters()

    @property
    def embedding_dim(self):
        """Return embedding dimension"""
        return self.nhid