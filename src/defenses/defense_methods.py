"""
Defense Methods for Byzantine-Robust Federated Learning.

This module implements various defense mechanisms against adversarial attacks
in federated learning:
- Krum: Distance-based client selection
- Flame: Clustering-based filtering with differential privacy
- Multi-Metrics: Combined distance metrics for outlier detection
- FedDMC: Dynamic Model Clustering
- FLPruning: Gradient pruning
"""

import torch
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod


class BaseDefense(ABC):
    """Abstract base class for defense mechanisms."""
    
    @abstractmethod
    def aggregate(
        self, 
        client_params: List[OrderedDict], 
        client_sizes: List[int]
    ) -> OrderedDict:
        """
        Aggregate client parameters with defense mechanism.
        
        Args:
            client_params: List of client model parameters
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated global model parameters
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Defense method name."""
        pass


class KrumDefense(BaseDefense):
    """
    Krum Defense.
    
    Selects the client whose parameters have the smallest sum of 
    Euclidean distances to the closest n-f-2 clients, where f is
    the number of Byzantine clients.
    
    Reference: Blanchard et al., "Machine Learning with Adversaries: 
               Byzantine Tolerant Gradient Descent", NeurIPS 2017
    """
    
    def __init__(self, num_byzantine: int = 1):
        self.num_byzantine = num_byzantine
    
    @property
    def name(self) -> str:
        return "Krum"
    
    def aggregate(
        self, 
        client_params: List[OrderedDict], 
        client_sizes: List[int]
    ) -> OrderedDict:
        n_clients = len(client_params)
        n_closest = max(1, n_clients - self.num_byzantine - 2)
        
        # Compute pairwise Euclidean distances
        distances = torch.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = 0
                for key in client_params[0].keys():
                    diff = client_params[i][key] - client_params[j][key]
                    dist += torch.norm(diff).item() ** 2
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute Krum scores
        scores = []
        for i in range(n_clients):
            dists = distances[i].clone()
            dists[i] = float('inf')
            closest_dists = torch.topk(
                dists, 
                min(n_closest, n_clients - 1), 
                largest=False
            )[0]
            scores.append(closest_dists.sum().item())
        
        # Select client with smallest score
        selected = np.argmin(scores)
        return client_params[selected]


class FlameDefense(BaseDefense):
    """
    Flame Defense.
    
    Uses clustering to identify and filter out anomalous updates,
    then adds differential privacy noise to the aggregation.
    
    Reference: Nguyen et al., "FLAME: Taming Backdoors in 
               Federated Learning", USENIX Security 2022
    """
    
    def __init__(self, noise_scale: float = 0.001, clip_bound: float = 2.0):
        self.noise_scale = noise_scale
        self.clip_bound = clip_bound
    
    @property
    def name(self) -> str:
        return "Flame"
    
    def aggregate(
        self, 
        client_params: List[OrderedDict], 
        client_sizes: List[int]
    ) -> OrderedDict:
        n_clients = len(client_params)
        
        # Flatten parameters for clustering
        flattened = []
        for params in client_params:
            flat = torch.cat([p.flatten() for p in params.values()])
            flattened.append(flat.cpu().numpy())
        
        # Cluster clients
        try:
            kmeans = KMeans(
                n_clusters=min(3, n_clients), 
                random_state=42, 
                n_init=10
            )
            labels = kmeans.fit_predict(flattened)
            
            # Select largest cluster (assumed to be honest)
            unique_labels, counts = np.unique(labels, return_counts=True)
            main_cluster = unique_labels[np.argmax(counts)]
            selected_indices = np.where(labels == main_cluster)[0]
        except Exception:
            selected_indices = list(range(n_clients))
        
        # Aggregate selected clients
        selected_params = [client_params[i] for i in selected_indices]
        selected_sizes = [client_sizes[i] for i in selected_indices]
        
        total_size = sum(selected_sizes)
        global_params = OrderedDict()
        
        for key in client_params[0].keys():
            global_params[key] = torch.zeros_like(client_params[0][key])
        
        for params, size in zip(selected_params, selected_sizes):
            weight = size / total_size
            for key in params.keys():
                global_params[key] += params[key] * weight
        
        # Add differential privacy noise
        for key in global_params.keys():
            noise = torch.randn_like(global_params[key]) * self.noise_scale
            global_params[key] += noise
        
        return global_params


class MultiMetricsDefense(BaseDefense):
    """
    Multi-Metrics Defense.
    
    Combines Manhattan, Euclidean, and Cosine distances to
    identify and exclude outlier clients.
    """
    
    def __init__(self, num_selected: int = 3):
        self.num_selected = num_selected
    
    @property
    def name(self) -> str:
        return "MultiMetrics"
    
    def aggregate(
        self, 
        client_params: List[OrderedDict], 
        client_sizes: List[int]
    ) -> OrderedDict:
        n_clients = len(client_params)
        
        # Compute combined distance metric for each client
        scores = []
        for i in range(n_clients):
            total_score = 0
            for j in range(n_clients):
                if i != j:
                    manhattan = 0
                    euclidean = 0
                    cos_num = 0
                    cos_den_i = 0
                    cos_den_j = 0
                    
                    for key in client_params[0].keys():
                        diff = client_params[i][key] - client_params[j][key]
                        manhattan += torch.abs(diff).sum().item()
                        euclidean += torch.norm(diff).item() ** 2
                        
                        cos_num += (client_params[i][key] * client_params[j][key]).sum().item()
                        cos_den_i += (client_params[i][key] ** 2).sum().item()
                        cos_den_j += (client_params[j][key] ** 2).sum().item()
                    
                    euclidean = np.sqrt(euclidean)
                    cos_dist = 1 - (cos_num / (np.sqrt(cos_den_i * cos_den_j) + 1e-8))
                    
                    combined = manhattan + euclidean + cos_dist
                    total_score += combined
            
            scores.append(total_score / max(1, n_clients - 1))
        
        # Select clients with smallest dispersion
        selected_indices = np.argsort(scores)[:min(self.num_selected, n_clients)]
        
        selected_params = [client_params[i] for i in selected_indices]
        selected_sizes = [client_sizes[i] for i in selected_indices]
        
        total_size = sum(selected_sizes)
        global_params = OrderedDict()
        
        for key in client_params[0].keys():
            global_params[key] = torch.zeros_like(client_params[0][key])
        
        for params, size in zip(selected_params, selected_sizes):
            weight = size / total_size
            for key in params.keys():
                global_params[key] += params[key] * weight
        
        return global_params


class FedDMCDefense(BaseDefense):
    """
    FedDMC: Federated Dynamic Model Clustering.
    
    Clusters client updates and uses the largest cluster
    for aggregation, filtering out potential adversaries.
    """
    
    def __init__(self, num_clusters: int = 3):
        self.num_clusters = num_clusters
    
    @property
    def name(self) -> str:
        return "FedDMC"
    
    def aggregate(
        self, 
        client_params: List[OrderedDict], 
        client_sizes: List[int]
    ) -> OrderedDict:
        n_clients = len(client_params)
        
        # Flatten parameters
        flattened = []
        for params in client_params:
            flat = torch.cat([p.flatten() for p in params.values()])
            flattened.append(flat.cpu().numpy())
        
        # Cluster
        kmeans = KMeans(
            n_clusters=min(self.num_clusters, n_clients), 
            random_state=42, 
            n_init=10
        )
        labels = kmeans.fit_predict(flattened)
        
        # Select largest cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        main_cluster = unique_labels[np.argmax(counts)]
        selected_indices = np.where(labels == main_cluster)[0]
        
        selected_params = [client_params[i] for i in selected_indices]
        selected_sizes = [client_sizes[i] for i in selected_indices]
        
        total_size = sum(selected_sizes)
        global_params = OrderedDict()
        
        for key in client_params[0].keys():
            global_params[key] = torch.zeros_like(client_params[0][key])
        
        for params, size in zip(selected_params, selected_sizes):
            weight = size / total_size
            for key in params.keys():
                global_params[key] += params[key] * weight
        
        return global_params


class FLPruningDefense(BaseDefense):
    """
    FLPruning Defense.
    
    Prunes small gradient values after aggregation to remove
    potential backdoor signals hidden in small updates.
    """
    
    def __init__(self, prune_ratio: float = 0.2):
        self.prune_ratio = prune_ratio
    
    @property
    def name(self) -> str:
        return "FLPruning"
    
    def aggregate(
        self, 
        client_params: List[OrderedDict], 
        client_sizes: List[int]
    ) -> OrderedDict:
        total_size = sum(client_sizes)
        global_params = OrderedDict()
        
        for key in client_params[0].keys():
            # Weighted aggregation
            aggregated = torch.zeros_like(client_params[0][key])
            for params, size in zip(client_params, client_sizes):
                weight = size / total_size
                aggregated += params[key] * weight
            
            # Prune small values
            flat = aggregated.flatten()
            k = int(len(flat) * self.prune_ratio)
            if k > 0:
                threshold = torch.topk(torch.abs(flat), k, largest=False)[0][-1]
                mask = torch.abs(aggregated) > threshold
                aggregated = aggregated * mask.float()
            
            global_params[key] = aggregated
        
        return global_params


def get_defense(
    defense_method: str, 
    num_clients: int = 5, 
    adversarial_ratio: float = 0.4,
    **kwargs
) -> BaseDefense:
    """
    Factory function to create defense instances.
    
    Args:
        defense_method: Name of the defense method
        num_clients: Total number of clients
        adversarial_ratio: Fraction of adversarial clients
        **kwargs: Additional defense-specific parameters
        
    Returns:
        Defense instance
    """
    n_byzantine = int(num_clients * adversarial_ratio)
    
    defenses = {
        'krum': lambda: KrumDefense(
            num_byzantine=kwargs.get('num_byzantine', n_byzantine)
        ),
        'flame': lambda: FlameDefense(
            noise_scale=kwargs.get('noise_scale', 0.001),
            clip_bound=kwargs.get('clip_bound', 2.0)
        ),
        'multi_metrics': lambda: MultiMetricsDefense(
            num_selected=kwargs.get('num_selected', 3)
        ),
        'feddmc': lambda: FedDMCDefense(
            num_clusters=kwargs.get('num_clusters', 3)
        ),
        'flpruning': lambda: FLPruningDefense(
            prune_ratio=kwargs.get('prune_ratio', 0.2)
        )
    }
    
    if defense_method not in defenses:
        raise ValueError(
            f"Unknown defense: {defense_method}. "
            f"Available: {list(defenses.keys())}"
        )
    
    return defenses[defense_method]()
