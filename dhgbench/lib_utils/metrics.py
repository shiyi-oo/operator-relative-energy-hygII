"""Node-classification accuracy used for epoch selection and reporting."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


@torch.no_grad()
def evaluate(model, data, split_idx, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out,_ = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_acc(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_acc(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_acc(
        data.y[split_idx['test']], out[split_idx['test']])
    
    return train_acc, valid_acc, test_acc


def masked_accuracy(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()


def accuracy(logits: Tensor, labels: Tensor, masks: dict[Tensor]):
    accs = []
    for mask in masks.values():
        acc = masked_accuracy(logits[mask], labels[mask])
        accs.append(acc)
    return accs

