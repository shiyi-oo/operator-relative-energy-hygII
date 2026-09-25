"""Full-batch node training with validation-selected epochs."""
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from lib_utils.metrics import evaluate
from lib_models import _semi_methods_


class Trainer:
    def __init__(self, args, **kwargs):
        """
        Training pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
        self.train_time = None
        self.best_val_acc = None
        self.best_epoch = None


    def semi_node_cls_training(self,model,data,masks,args):
        
        criterion = nn.NLLLoss()
        
        model.train()
        
        ### Training loop ###
        start_time = time.time()
        model.reset_parameters()
        
        if args.method in ('UniGCNII', 'UniGCN'):
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        best_val_acc = -1
        best_epoch = 0
        best_model = copy.deepcopy(model)
        
        #for epoch in tqdm(range(args.epochs)):
        for epoch in trange(args.epochs,disable=True):
        # for epoch in range(args.epochs):
            # Training part
            model.train()
            optimizer.zero_grad()
            out, _ = model(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[masks['train']], data.y[masks['train']])
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_thresh)
            optimizer.step()

            result = evaluate(model, data, masks)
            if result[1] >= best_val_acc:
                best_val_acc = result[1]
                best_epoch = epoch + 1
                best_model = copy.deepcopy(model)

            # if (epoch+1) % args.display_step == 0:
            #     print(f'Epoch: {epoch+1:02d}, '
            #         f'Train Acc: {100 * result[0]:.2f}%, '
            #         f'Valid Acc: {100 * result[1]:.2f}%, '
            #         f'Test  Acc: {100 * result[2]:.2f}%')

                
        end_time = time.time()
        self.train_time = end_time-start_time
        self.best_val_acc = best_val_acc
        self.best_epoch = best_epoch
        
        
        return best_model


    def training(self, model, data, args, seed_split=None, task_type='node_cls'):
        if task_type != 'node_cls' or args.method not in _semi_methods_:
            raise ValueError(f'Unsupported study task/model: {task_type}/{args.method}')
        return self.semi_node_cls_training(model, data, seed_split, args)
