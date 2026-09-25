import random
import numpy as np
import torch

def mean_std_metrics(metrics):
    
    metrics = np.array(metrics)
    metrics_dim = metrics.shape[-1]
    metrics = metrics.reshape(-1, metrics_dim) 
    
    metrics_mean = list(np.mean(metrics, axis=0))
    metrics_std = list(np.std(metrics, axis=0))
    
    return metrics_mean, metrics_std


def result_printer(metrics,name):
    
    metrics_dim = np.array(metrics).shape[-1]
    metrics_mean,metrics_std = mean_std_metrics(metrics)
    
    if metrics_dim == 1:
        print(f'{name}: {metrics_mean[0]:.4f}+-{metrics_std[0]:.2f}')
    else:
        print(f'train_{name}: {metrics_mean[0]:.4f}+-{metrics_std[0]:.2f}, val_{name}: {metrics_mean[1]:.4f}+-{metrics_std[1]:.2f}, test_{name}: {metrics_mean[2]:.4f}+-{metrics_std[2]:.2f}')


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

