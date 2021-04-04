import argparse
import sys
from itertools import product
import pandas as pd

def get_args(argv):
    
    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()
    
    # stream vs offline learning
    parser.add_argument('--offline', default = False, action = 'store_true', dest = 'offline', help = "offline vs online (stream learning) training")

    # scenario/task
    parser.add_argument('--scenario', nargs="+", default = ['iid', 'instance', 'class_iid', 'class_instance'], 
                        help = "How to set up tasks, e.g. iid => randomly assign data to each task")
    parser.add_argument('--n_runs', type = int, default = 1, help = "Number of times to repeat the experiment with different data orderings")
    
    # model hyperparameters/type
    parser.add_argument('--model_type', type=str, default='resnet', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='ResNet18', help="The name of actual model for the backbone")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--batch_size', nargs="+", default=[100], type=int)
    parser.add_argument('--lr', nargs="+", default=[0.01], type=float, help="Learning rate")
    parser.add_argument('--momentum', nargs="+", default=[0], type=float)
    parser.add_argument('--weight_decay', nargs="+", default=[0], type=float)
    parser.add_argument('--pretrained', default=False, dest='pretrained', action = 'store_true')
    parser.add_argument('--freeze_feature_extract', default = False, dest = 'freeze_feature_extract', action = 'store_true')
    parser.add_argument('--n_epoch', nargs="+", default = [1], type=int, help="Number of epochs to train")
    
    # for regularization models
    parser.add_argument('--reg_coef', nargs="+", default=[1], type=float, help="The coefficient for regularization. Larger means less plasilicity. ")

    # for replay models
    parser.add_argument('--memory_size', nargs="+", default=[1200], type=int, help="Number of training examples to keep in memory")

    # directories
    parser.add_argument('--dataroot', type = str, default = 'data/core50', help = "Directory that contains the data")
    parser.add_argument('--filelist_root', type = str, default = 'dataloaders', help = "Directory that contains the filelists for each task")
    parser.add_argument('--dataset', type = str, default = 'core50', help = "Which dataset to load")
    parser.add_argument('--output_dir', default='outputs',
                        help="Where to store accuracy table")
    
    # gpu/cpu settings
    parser.add_argument('--gpuid', nargs="+", default=[0,1,2,3], type=int,
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--n_workers', default=30, type = int, help="Number of cpu workers for dataloader")
    
    # filename to store gridsearch in
    parser.add_argument('--grid_name', type = str, default='grid00', help="Name for the .csv file the gridsearch will be stored in")
    
    # return parsed arguments
    args = parser.parse_args(argv)
    return args


def main():
    
    # get command line arguments
    args = get_args(sys.argv[1:])
    
    # convert argument namespace to dictionary
    args = vars(args)
    # get filename, remove from args since its not a model parameter
    # do the same for the gpuids
    fname = args['grid_name']
    del args['grid_name']
    gpuid = args['gpuid']
    del args['gpuid']
    
    # encasing each value of args in a list if not already a list 
    # enables iteration over values
    for k, v in args.items():
        if type(v) != list:
            args[k] = [v]
            
    print(args)
    
    # get a cartesian product of the parameters
    arg_grid = pd.DataFrame.from_dict([dict(zip(args.keys(), v)) for v in product(*args.values())])
    
    # assign each run to one of the available gpus
    n_gpu = len(gpuid)
    arg_grid['gpuid'] = -1
    gpu_col = arg_grid.columns.get_loc('gpuid')
    for i in range(arg_grid.shape[0]):
        # mod i by n gpus
        g = i % n_gpu
        arg_grid.iloc[i, gpu_col] = gpuid[g]
    
    # write to csv
    name = './gridsearches/' + fname + '.csv'
    arg_grid.to_csv(name)


if __name__ == '__main__':
    
    main()
