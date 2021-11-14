import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math


def get_args(argv):
    
    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()

    # plotting parameters
    parser.add_argument('--scenario', type = str, default = 'iid', help = "How to set up tasks, e.g. iid => randomly assign data to each task")
    parser.add_argument('--output_dir', default='outputs',
                        help="Where to store accuracy table")
    parser.add_argument('--result_dir', default=['NormalNN_ResNet18'], nargs="+", help="a custom subdirectory to store results")
    parser.add_argument('--validation', default=False, action='store_true',  dest='validate', help="Plot validation accuracy instead of testing")
    parser.add_argument('--n_class_per_task', type=int, default=2, help="Number of classes trained on in each task")
    parser.add_argument('--task1', default=False, action='store_true', dest='task1', help="Plot accuracy on task 1 instead of all tasks trained on so far")
    
    # return parsed arguments
    args = parser.parse_args(argv)
    return args


def main():
    
    # get command line arguments
    args = get_args(sys.argv[1:])
    
    # appending path to cwd to directories
    args.output_dir = os.path.join(os.getcwd(),args.output_dir)
    
    # ensure that a valid scenario has been passed
    if args.scenario not in ['iid', 'class_iid', 'instance', 'class_instance']:
        print('Invalid scenario passed, must be one of: iid, class_iid, instance, class_instance')
        return
    
    fig, ax = plt.subplots()
    names=[]
    
    for r in args.result_dir:
        path = os.path.join(args.output_dir, args.scenario, r)
        if args.validate:
            if args.task1:
                raise ValueError("Cannot plot task1 accuracy results for validation set - data not available")
            else:
                try:
                    result = pd.read_csv(os.path.join(path, 'val_all_mem_all_runs.csv'), header = None)
                except FileNotFoundError:
                    # default to old naming scheme
                    result = pd.read_csv(os.path.join(path, 'val_task1.csv'), header = None)
        else:
            if args.task1:
                try:
                    result = pd.read_csv(os.path.join(path, 'test_1st_mem_all_runs.csv'), header=None)
                except FileNotFoundError:
                    # default to old naming scheme
                    result = pd.read_csv(os.path.join(path, 'test_task1.csv'), header=None)
            else:
                try:
                    result = pd.read_csv(os.path.join(path, 'test_all_mem_all_runs.csv'), header=None)
                except FileNotFoundError:
                    # default to old naming scheme
                    result = pd.read_csv(os.path.join(path, 'test.csv'), header=None)
        hyperparams = pd.read_csv(os.path.join(path, 'hyperparams.csv'), header = None, error_bad_lines=False, index_col=0)

        x = result.columns
        y = result.mean(axis=0)
        yerr = result.std(axis=0) / math.sqrt(len(result.index))
        name = hyperparams.loc['agent_name'].values[0]
        names.append(name)

        ax.errorbar(x,y,yerr, label = name, capsize=2)

    if args.scenario in ['class_iid', 'class_instance']:
        x = result.columns
        y = [100/((task+1)*args.n_class_per_task) for task in range(len(x))]
        ax.plot(x,y, label='Chance')

    ax.legend()
    ax.set_ylabel('Accuracy')
    ax.set_yticks([t for t in range(0,100,10)])
    ax.set_xlabel('Task')
    ax.set_xticks([t for t in range(len(result.columns))])
    ax.set_xticklabels([t+1 for t in range(len(result.columns))]) # Task labelling starts at 1, not 0

    if args.task1:
        ax.set_title(args.scenario + "\n Task 1 accuracy")
    else:
        ax.set_title(args.scenario + "\n Accuracy on seen classes")
    
    fname = '-'.join(names) + '_' + args.scenario
    if args.task1:
        fname = fname + "_task1"
    
    fig.savefig(os.path.join('plots', fname), dpi=300)

if __name__ == '__main__':
    
    main()
    
    
    