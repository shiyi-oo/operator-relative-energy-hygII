"""Node-classification evaluation for the study models."""
import torch
from lib_utils.metrics import accuracy


class Evaluator:
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = args.device

    @torch.no_grad()
    def node_cls_evaluation(self, model, data, masks, result=None):

        if result is not None:
            logits = result
        else:
            model.eval()
            logits,_ = model(data) 

        accs = accuracy(logits, data.y, masks)
        metrics_dict = {'acc': accs}

        return metrics_dict


    def evaluate(self, model, data, seed_split=None, task_type='node_cls', verbose=False):
        if task_type != 'node_cls':
            raise ValueError('Only node classification is included in these studies')
        metrics = self.node_cls_evaluation(model, data, seed_split)
        if verbose:
            for name, values in metrics.items():
                print(f'train_{name}: {values[0]:.2f}, valid_{name}: {values[1]:.2f}, test_{name}: {values[2]:.2f} ')
        return metrics
