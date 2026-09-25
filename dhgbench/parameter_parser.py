import argparse
import os
import sys
import yaml
from lib_dataset import _single_datasets_
from lib_models import _semi_methods_

def cli_specified_keys(argv=None, parser=None):
    """Dest names of the options explicitly given on the command line.

    argparse cannot tell "user passed the default value" from "user passed nothing",
    so we read argv directly. Used to keep an explicit --flag from being silently
    overwritten by the YAML config (which is what made hyperparameter sweeps that
    combine --use_yaml with an explicit --lr a no-op).

    Only tokens that match a real option string count, so an option VALUE that happens
    to start with '--' is not mistaken for a flag, and everything after a bare '--'
    terminator is ignored. The parser is built with allow_abbrev=False, so a token
    always maps to exactly one dest -- with abbreviations enabled, '--restart' would
    set restart_alpha while going undetected here, and the YAML would silently win.
    """
    argv = sys.argv[1:] if argv is None else argv
    known = {}
    if parser is not None:
        for action in parser._actions:
            for opt in action.option_strings:
                known[opt] = action.dest
    keys = set()
    for tok in argv:
        if tok == '--':
            break
        if not tok.startswith('-') or tok == '-':
            continue
        name = tok.split('=', 1)[0]
        if known:
            if name in known:
                keys.add(known[name])
        else:
            keys.add(name.lstrip('-').replace('-', '_'))
    return keys

def update_from_dict(obj, updates, protected=()):
    for key, value in updates.items():
        # an explicit command-line value always wins over the YAML config
        if key in protected:
            continue
        # set higher priority from command line as we explore some factors
        if key in ['init'] and getattr(obj, 'init', None) is not None:
            continue
        setattr(obj, key, value)

# recommend hyperparameters here
def method_config(args, protected=None, parser=None):

    if protected is None:
        protected = cli_specified_keys(parser=parser or getattr(args, '_parser', None))

    if args.is_default:
        config_name = 'default'
    else:
        config_name = args.dname
    try:
        # conf_dt = json.load(open(f"{os.path.join('./', 'lib_configs', args.method.lower(), config_name)}.json")) 
        task_prefix=args.task_type.split('_')[0]+'_yamls'
        conf_dt = yaml.safe_load(open(f"{os.path.join('./', 'lib_yamls', task_prefix,'config_'+args.method.lower())}.yaml"))[config_name] 
        update_from_dict(args, conf_dt, protected=protected)
        if protected:
            kept = sorted(k for k in protected if k in conf_dt)
            if kept:
                print(f'CLI overrides kept over YAML: {kept}')
    except Exception as e:
        # Before the model hyperparameters became real CLI arguments, a missing or broken
        # YAML crashed the run with AttributeError. Now generic defaults quietly apply, so
        # this has to be loud: those defaults are not the published settings and can be
        # materially wrong (tens of accuracy points on some method/dataset pairs).
        print('=' * 78)
        print(f'WARNING: failed to load YAML config for method={args.method}, '
              f'task_type={args.task_type}, dataset={config_name}: {e}')
        print('WARNING: falling back to generic argparse defaults for the model '
              'hyperparameters. These are NOT the published settings.')
        print('=' * 78)

    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_task_args(args, protected=None):

    if protected is None:
        protected = getattr(args, '_cli_keys', set())

    if not args.use_yaml:
        print('=' * 78)
        print('WARNING: --use_yaml was not passed. Model hyperparameters come from the '
              'generic argparse defaults,')
        print('WARNING: which are NOT the published per-method/per-dataset settings.')
        print('=' * 78)

    if args.task_type == 'node_cls':
        if args.dname not in _single_datasets_:
            raise ValueError('The dataset is not suitable for node classification')
        args.add_self_loop=True 
        if args.use_bench_prop and not ({'train_prop','valid_prop'} & set(protected)):
            args.train_prop,args.valid_prop = 0.5,0.25
        if 'early_stop' not in protected:
            args.early_stop = False
    else:
        raise ValueError('Only node classification is included in these studies')

    return args

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a good quality representation without grid search.
    """
    # allow_abbrev=False: an abbreviated flag would still be applied by argparse but
    # would go undetected by cli_specified_keys, letting the YAML silently overwrite it.
    parser = argparse.ArgumentParser(allow_abbrev=False)

    ######################### general parameters ################################
    '''
    Semi-supervised setting: Train/Valid/Test: 50/25/25
    
    '''
    parser.add_argument('--use_bench_prop', default=True)
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.2)

    parser.add_argument('--dname', default='cora', choices=_single_datasets_)
    parser.add_argument('--task_type', default='node_cls', choices=['node_cls'])
    parser.add_argument('--is_default', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--method', default='HGNN', choices=_semi_methods_) 
    
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_seeds', type=int, default=20)
    parser.add_argument('--epochs', default=200, type=int) 
    parser.add_argument("--All_num_layers", default=2, type=int)
    parser.add_argument("--MLP_hidden", default=128, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--use_yaml', action='store_true')

    parser.add_argument('--clip_grad',default=False,type=bool)
    parser.add_argument('--clip_thresh',default=5.0,type=float)
    parser.add_argument('--display_step', type=int, default=20)
    parser.add_argument('--eval_verbose',default=True,type=bool)
    
    parser.add_argument('--embedding_mode',default=True,type=bool) 
    parser.add_argument('--embedding_hidden',default=128,type=int) 
    
    parser.add_argument('--normtype', default='all_one') # ['all_one','deg_half_sym']
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--exclude_self', action='store_true')
    
    parser.add_argument('--early_stop', default=True)

    parser.add_argument('--tune_seed', default=False, action='store_true')

    ## methods args
    # HGNN
    parser.add_argument('--HCHA_symdegnorm', action='store_true')

    # ---- II-variant hyperparameters (shared by HGNNII/HNHNII/HyperGCNII/
    # ---- AllSetformerII/UniGCNII). Previously YAML-only, which made
    # ---- them untunable from the command line.
    parser.add_argument('--restart_alpha', type=float, default=0.1,
                        help='alpha in X <- (1-alpha) P X + alpha X^(0)')
    parser.add_argument('--lamda', type=float, default=0.5,
                        help='lamda in beta_l = log(lamda/l + 1), W_beta = (1-beta)I + beta W')

    # HNHN
    parser.add_argument('--HNHN_alpha', type=float, default=-1.5)
    parser.add_argument('--HNHN_beta', type=float, default=-0.5)
    parser.add_argument('--HNHN_nonlinear_inbetween', type=str2bool, nargs='?', const=True, default=True,
                        help='published inner ReLU of the base HNHN model')
    parser.add_argument('--HNHN_II_nonlinear_inbetween', type=str2bool, nargs='?', const=True, default=False,
                        help='inner ReLU for HNHNII; off keeps P a fixed linear operator')

    # HyperGCN
    parser.add_argument('--HyperGCN_mediators', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--HyperGCN_fast', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--HyperGCN_lazy', type=str2bool, nargs='?', const=True, default=True,
                        help='HyperGCNII on the lazy operator (I+A)/2, so spec(P) in [0,1]')
    parser.add_argument('--HyperGCN_zero_bias', type=str2bool, nargs='?', const=True, default=True,
                        help='zero-init the layer bias (a nonzero constant masks over-smoothing)')

    # AllSetTransformer
    parser.add_argument('--MLP_num_layers', type=int, default=2)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'add', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn', 'ln', 'None'])
    parser.add_argument('--deepset_input_norm', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--GPR', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--LearnMask', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--PMA', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--output_heads', type=int, default=1)
    parser.add_argument('--decoder_hidden', type=int, default=64)
    parser.add_argument('--decoder_num_layers', type=int, default=2)
    parser.add_argument('--InputNorm', type=str2bool, nargs='?', const=True, default=False)

    # UniGCN / UniGCNII
    parser.add_argument('--input_drop', type=float, default=0.6)
    parser.add_argument('--activation', default='relu', choices=['relu', 'prelu'])
    parser.add_argument('--use_norm', type=str2bool, nargs='?', const=True, default=False)

    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='0.6', type=str)
    
    parser.set_defaults(add_self_loop=False)
    parser.set_defaults(exclude_self=False)

    args = parser.parse_args()
    # stash for method_config / set_task_args / exp_agent so that "the user asked for this
    # explicitly" survives every later mutation of args
    args._cli_keys = cli_specified_keys(parser=parser)
    args._parser = parser

    return args
