"""Model dispatch and repeated-seed node classification."""
from collections import defaultdict
import numpy as np
from lib_utils.utils import fix_seed, result_printer, mean_std_metrics
from lib_utils.train_agent import Trainer
from lib_utils.eval_agent import Evaluator
from lib_models.HNN import (HCHA, HCHAII, HyperGCN, HyperGCNII,
                            SetGNN, SetGNNII, UniGCN, UniGCNII)

from lib_models.HNN.hnhn_matched import MatchedHNHN


class ExpAgent:
    def __init__(self,args,**kwargs):
        """
        Overall pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
        self.trainer=Trainer(args)
        self.evaluator=Evaluator(args)
        self.train_times = []


    def node_cls_train_eval(self,data):
        
        metrics_dict=defaultdict(list)

        for seed in range(self.args.num_seeds):
            if self.args.tune_seed:
                seed += 100
            fix_seed(seed) 
            
            masks=data.generate_random_split(train_ratio=self.args.train_prop,val_ratio=self.args.valid_prop,seed=seed)

            model = parse_model(self.args, data).to(self.args.device)

            model = self.trainer.training(model,data,self.args,seed_split=masks,task_type='node_cls')
            
            self.train_times.append(self.trainer.train_time)

            if self.args.eval_verbose:
                print(f'[Seed {seed}]: ', end='')
                result=self.evaluator.evaluate(model,data,seed_split=masks,task_type='node_cls',verbose=True)
                print(
                    f'train_time:{self.trainer.train_time:.2f}; '
                    f'best_val_acc:{self.trainer.best_val_acc:.2f} at epoch {self.trainer.best_epoch:02d}'
                )
            else:
                result=self.evaluator.evaluate(model,data,seed_split=masks,task_type='node_cls',verbose=False)
            
            for m in result:
                metrics_dict[m].append(result[m])
            
        print(f'[Final]: ', end='')
        self.test_dict = defaultdict(list) 
        for m in metrics_dict:
            result_printer(metrics_dict[m],m)
            metrics_mean, metrics_std = mean_std_metrics(metrics_dict[m])
            self.test_dict[m].extend([metrics_mean[-1],metrics_std[-1]])
        print(f'avg_train_time:{np.mean(self.train_times):2f}')
        print(f'------------------------------------------------------------------------------')


    def running(self, task_type, data):
        if task_type != 'node_cls':
            raise ValueError('Only node classification is included in these studies')
        self.node_cls_train_eval(data)


def parse_model(args, data):
    num_targets = args.embedding_hidden if args.embedding_mode else data.num_classes
    if args.method in ('HNHN', 'HNHNII'):
        return MatchedHNHN(data.num_features, data.num_classes, args, use_ii=args.method == 'HNHNII')
    if args.method in ('AllSetformer', 'AllSetformerII'):
        model = SetGNN if args.method == 'AllSetformer' else SetGNNII
        if args.LearnMask:
            return model(data.num_features, num_targets, args, data.norm)
        return model(data.num_features, num_targets, args)
    models = {'HGNN': HCHA, 'HGNNII': HCHAII,
              'HyperGCN': HyperGCN, 'HyperGCNII': HyperGCNII,
              'UniGCN': UniGCN, 'UniGCNII': UniGCNII}
    if args.method not in models:
        raise ValueError(f'Unsupported study model: {args.method}')
    return models[args.method](data.num_features, num_targets, args)
