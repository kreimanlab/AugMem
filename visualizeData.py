import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import sys
import argparse
from dataloaders import datasets
import matplotlib as mpl

# ad hoc, ineffecient method of getting two factors closest to sqr root of n
def getClosestFactors(n):

    val = int(math.sqrt(n))

    while (n % val != 0):
        val -= 1

    return (n/val, val)

# code for visualizing dataset
def visualize(data, inds, scenario, task, outdir):

    mpl.rcParams.update({'font.size': 8})

    fig, ax = plt.subplots()
    fig.suptitle("Task " + str(task))

    rows, cols = getClosestFactors(len(inds))

    for i in range(len(inds)):

        im, label = data[inds[i]]
        plt.subplot(rows, cols, i+1)
        plt.imshow(im)
        plt.title("S{}, C{}".format(inds[i], label))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

    if not os.path.exists(outdir + '/' + scenario + '/task_images'):
        os.makedirs(outdir + '/' + scenario + '/task_images')

    fig.savefig(outdir + '/' + scenario + '/task_images/task' + str(task) + '_images.png', dpi = 300)
    plt.close(fig)


def get_args(argv):

    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()

    # scenario/task
    parser.add_argument('--scenario', type = str, default = 'iid', help = "Which task setup to visualize, e.g. iid => randomly assign data to each task")
    parser.add_argument('--run', type = int, default = 0, help = "Which run to visualize images from")

    # display options
    parser.add_argument('--im_start', type = int, default = 0, help = "index of first image to show")
    parser.add_argument('--n_image', type = int, default = 4, help = "How many images to display at a time")
    parser.add_argument('--increment', type = int, default = 0, help = "Display --n_image images every increment number of images; zero means don't increment, must be greater than n_image")
    parser.add_argument('--n_increment', type = int, default = -1, help = "Number of times to increment")

    # directories
    parser.add_argument('--dataroot', type = str, default = '../data/core50/', help = "Directory that contains the data")
    parser.add_argument('--output_dir', default='data_viz',
                        help="Where to store accuracy table")

    # return parsed arguments
    args = parser.parse_args(argv)
    return args

def main():

    # get command line arguments
    args = get_args(sys.argv[1:])

    # append path to cwd to directories
    args.dataroot = os.path.join(os.getcwd(),args.dataroot)
    args.output_dir = os.path.join(os.getcwd(),args.output_dir)

    # read dataframe containing information for each task
    task_df = pd.read_csv(os.path.join(args.dataroot, 'task_filelists', args.scenario, 'run' + str(args.run), 'train_all.txt'), index_col = 0)

    tasks = task_df.task.unique()

    for task in tasks:
        # get training data pertaining to chosen scenario, task, run
        train_data = datasets.CORE50(
                root = args.dataroot, scenario = args.scenario, run = args.run, batch = task)

        if args.increment == 0:
            inds = list(range(args.im_start, args.im_start + args.n_image))
        else:
            assert args.increment > args.n_image, "n_increment should be greater than n_image, or zero to denote that no incrementing should be done"

            if args.n_increment < 0:
                inds = [list(range(i, i + args.n_image)) for i in range(args.im_start, len(train_data), args.increment)]
                inds = [item for sublist in inds for item in sublist]
            else:
                inds = [list(range(i, i + args.n_image)) for i in range(args.im_start, min(len(train_data), args.im_start + args.increment*args.n_increment), args.increment)]
                inds = [item for sublist in inds for item in sublist]

        # print(inds)
        visualize(train_data, inds, args.scenario, task, args.output_dir)


if __name__ == '__main__':

    main()
