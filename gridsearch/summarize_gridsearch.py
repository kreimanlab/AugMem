import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt

def main():
    
    # get command line arguments
    args = sys.argv[1:]
    grid_dir = args[0]
    
    # get results for each experiment in the gridsearch
    results = {}
    for root, dirs, files in os.walk(grid_dir, topdown=False):
        for d in dirs:
            path = os.path.join(root, d, 'val.csv')
            result = pd.read_csv(path, header = None)
            results[d] = result
            
    # get the 10 experiments with the highest overall accuracies (across tasks, runs)
    means = {k:v.mean().mean() for k,v in results.items()}
    top = {k:means[k] for k in sorted(means, key=means.get, reverse=True)}
    print(top)
    top_df = pd.Series(top)
    top10 = list(top.keys())[:10]
    # filter the results for just these experiments
    results_10 = {k:v for k,v in results.items() if k in top10}
    
    avg_acc = {}
    
    # create line plot with std error bars
    for k, v in results_10.items():
        x = v.columns
        y = v.mean(axis=0)
        yerr = v.std(axis=0) / math.sqrt(len(v.index))
        plt.errorbar(x,y,yerr, label = k, capsize=2)
        avg_acc[k] = y.mean()
    
    best = top10[0]
    plt.legend(bbox_to_anchor=(1, .8))
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('Best: ' + best)
    plt.savefig(grid_dir + '/summary_plot.png', dpi = 300, bbox_inches='tight')
    top_df.to_csv(grid_dir + '/summary.csv')


if __name__ == '__main__':
    
    main()