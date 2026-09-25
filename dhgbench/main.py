"""Run one study model on a node-classification benchmark."""
from lib_utils.exp_agent import ExpAgent
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_dataset.data_base import HyperDataset
from lib_dataset.preprocessing import data_processing
from parameter_parser import parameter_parser, method_config, set_task_args


if __name__ == '__main__':
    args = parameter_parser()
    if args.use_yaml:
        args = method_config(args)
    args = set_task_args(args)
    data = data_processing(args, HyperDataset(args))
    data._initialization_()
    data = algo_preprocessing(data, args)
    print(f'All_num_layers: {args.All_num_layers}, lr: {args.lr}, wd: {args.wd}, MLP_hidden: {args.MLP_hidden}')
    ExpAgent(args).running(args.task_type, data)
