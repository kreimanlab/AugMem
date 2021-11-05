import os
import sys
import argparse
import random
import numpy as np
import torch
import pandas as pd
from dataloaders import datasets
from torchvision import transforms
import agents


def get_out_path(args):
    if args.custom_folder is None:
        if args.offline:
            subdir = args.agent_name + '_' + args.model_name + '_' + 'offline/'
        else:
            subdir = args.agent_name + '_' + args.model_name
    else:
        subdir = args.custom_folder

    total_path = os.path.join(args.output_dir, args.scenario, subdir)

    # make output directory if it doesn't already exist
    if not os.path.exists(total_path):
        os.makedirs(total_path)

    return total_path


def run(args, run):

    # read dataframe containing information for each task
    if args.offline:
        task_df = pd.read_csv(os.path.join('dataloaders', args.dataset + '_task_filelists', args.scenario, 'run' + str(run), 'offline', 'train_all.txt'), index_col = 0)
    else:
        task_df = pd.read_csv(os.path.join('dataloaders', args.dataset + '_task_filelists', args.scenario, 'run' + str(run), 'stream', 'train_all.txt'), index_col = 0)

    # get classes for each task
    active_out_nodes = task_df.groupby('task')['label'].unique().map(list).to_dict()

    # get tasks
    tasks = task_df.task.unique()

    # include classes from previous task in active output nodes for current task
    for i in range(1, len(tasks)):
        active_out_nodes[i].extend(active_out_nodes[i-1])

    # since the same classes might be in multiple tasks, want to consider only the unique elements in each list
    # mostly an aesthetic thing, will not affect results
    for i in range(1, len(tasks)):
        active_out_nodes[i] = list(set(active_out_nodes[i]))

    # agent parameters
    agent_config = {
        'lr': args.lr,
        'n_class': None,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'model_type' : args.model_type,
        'model_name' : args.model_name,
        'agent_type' : args.agent_type,
        'agent_name' : args.agent_name,
        'model_weights': args.model_weights,
        'pretrained': args.pretrained,
        'feature_extract' : False,
        'freeze_feature_extract': args.freeze_feature_extract,
        'optimizer':args.optimizer,
        'gpuid': args.gpuid,
        'reg_coef': args.reg_coef,
        'memory_size': args.memory_size,
        'n_workers' : args.n_workers,
        'memory_Nslots': args.memory_Nslots,
        'memory_Nfeat': args.memory_Nfeat,
        'freeze_batchnorm': args.freeze_batchnorm,
        'freeze_memory': args.freeze_memory,
        'batch_size': args.batch_size
        }

    if args.dataset == "core50":
        agent_config["n_class"] = 10
    elif args.dataset == "toybox":
        agent_config["n_class"] = 12
    elif args.dataset == "ilab2mlight":
        agent_config["n_class"] = 14
    elif args.dataset == "cifar100":
        agent_config["n_class"] = 100
    else:
        raise ValueError("Invalid dataset name, try 'core50', 'toybox', or 'ilab2mlight' or 'cifar100'")

    # initialize agent
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)

    if args.dataset == 'core50':
        # image transformations
        composed = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # get test data
        test_data = datasets.CORE50(
                    dataroot = args.dataroot, filelist_root = args.filelist_root, scenario = args.scenario, offline = args.offline, run = run, train = False, transform=composed)
    elif args.dataset == 'toybox' or args.dataset == 'ilab2mlight' or args.dataset == 'cifar100':
        # image transformations
        composed = transforms.Compose(
            [transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # get test data
        test_data = datasets.Generic_Dataset(
            dataroot=args.dataroot, dataset=args.dataset, filelist_root=args.filelist_root, scenario=args.scenario, offline=args.offline,
            run=run, train=False, transform=composed)
    else:
        raise ValueError("Invalid dataset name, try 'core50' or 'toybox' or 'ilab2mlight' or 'cifar100'")

    if args.validate:
        # splitting test set into test and validation
        test_size = int(0.75 * len(test_data))
        val_size = len(test_data) - test_size
        test_data, val_data = torch.utils.data.random_split(test_data, [test_size, val_size])
    else:
        val_data = None

    test_accs_1st, test_accs, val_accs, test_accs_all_epochs, test_accs_1st_all_epochs = train(agent, composed, args, run, tasks, active_out_nodes, test_data, val_data)

    return test_accs_1st, test_accs, val_accs, test_accs_all_epochs, test_accs_1st_all_epochs


def train(agent, transforms, args, run, tasks, active_out_nodes, test_data, val_data):

    if args.offline:
        print('============BEGINNING OFFLINE LEARNING============')
    else:
        print('============BEGINNING STREAM LEARNING============')

    # number of tasks
    ntask = len(tasks)

    # to store test accuracies
    test_accs = []
    test_accs_1st = []

    # to store val accuracies
    val_accs = []

    test_accs_all_epochs = []
    test_accs_1st_all_epochs = []
    val_accs_all_epochs = []

    # iterate over tasks
    for task in range(ntask):

        print('=============Training Task ' + str(task) + '=============')

        agent.active_out_nodes = active_out_nodes[task]

        print('Active output nodes for this task: ')
        print(agent.active_out_nodes)

        test_accs_all_epochs.append([])
        test_accs_1st_all_epochs.append([])
        val_accs_all_epochs.append([])

        if (args.n_epoch_first_task is not None) and (task == 0):
            n_epoch = args.n_epoch_first_task
        else:
            n_epoch = args.n_epoch
        for epoch in range(n_epoch):

            print('===' + args.agent_name + '; Epoch ' + str(epoch) + '; RUN ' + str(run) + '; TASK ' + str(task))

            # get training data pertaining to chosen scenario, task, run
            if args.dataset == 'core50':
                train_data = datasets.CORE50(
                    dataroot=args.dataroot, filelist_root=args.filelist_root, scenario=args.scenario,
                    offline=args.offline, run=run, batch=task, transform=transforms)
            elif args.dataset == 'toybox' or args.dataset == 'ilab2mlight' or args.dataset == 'cifar100':
                train_data = datasets.Generic_Dataset(
                    dataroot=args.dataroot, dataset=args.dataset, filelist_root=args.filelist_root, scenario=args.scenario,
                    offline=args.offline, run=run, batch=task, transform=transforms)
            else:
                raise ValueError("Invalid dataset name, try 'core50', 'toybox', or 'ilab2mlight' or 'cifar100'")

            # get train loader
            train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

            if args.validate:
                # then test and val data are subsets, not datasets and need to be dealt with accordingly
                # get test data only for the seen classes
                test_inds = [i for i in range(len(test_data)) if test_data.dataset.labels[test_data.indices[i]] in agent.active_out_nodes] # list(range(len(test_data)))
                task_test_data = torch.utils.data.Subset(test_data, test_inds)
                #labels = [task_test_data[i] for i in range(len(task_test_data))]
                test_loader = torch.utils.data.DataLoader(
                            task_test_data, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)
                val_inds = [i for i in range(len(val_data)) if val_data.dataset.labels[val_data.indices[i]] in agent.active_out_nodes]
                task_val_data = torch.utils.data.Subset(val_data, val_inds)
                val_loader = torch.utils.data.DataLoader(
                        task_val_data, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

            else:
                # get test data only for the seen classes
                test_inds = [i for i in range(len(test_data)) if test_data.labels[i] in agent.active_out_nodes] # list(range(len(test_data)))
                task_test_data = torch.utils.data.Subset(test_data, test_inds)
                test_loader = torch.utils.data.DataLoader(
                            task_test_data, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

                test_inds_1st = [i for i in range(len(test_data)) if test_data.labels[i] in active_out_nodes[0]] # retrive first task
                task_test_data_1st = torch.utils.data.Subset(test_data, test_inds_1st)
                test_loader_1st = torch.utils.data.DataLoader(
                            task_test_data_1st, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

            # learn
            agent.learn_stream(train_loader)

            # validate if applicable
            if args.validate:
                val_acc, val_time = agent.validation(val_loader)
                print(' * Val Acc: {acc:.3f}, Time: {time:.2f}'.format(acc=val_acc, time=val_time))
                val_accs_all_epochs[task].append(val_acc)

            test_acc, test_time = agent.validation(test_loader)
            print(' * Test Acc: {acc:.3f}, Time: {time:.2f}'.format(acc=test_acc, time=test_time))
            test_accs_all_epochs[task].append(test_acc)

            test_acc_1st, test_time_1st = agent.validation(test_loader_1st)
            print(' * Test Acc (1st): {acc:.3f}, Time: {time:.2f}'.format(acc=test_acc_1st, time=test_time_1st))
            test_accs_1st_all_epochs[task].append(test_acc_1st)

            if args.visualize:
                attread_filename = 'visualization/' + args.scenario + '/' + args.scenario + '_run_' + str(run) + '_task_' + str(task) + '_epoch_' + str(epoch)
                agent.visualize_att_read(attread_filename)
                agent.visualize_memory(attread_filename)

            if args.keep_best_net_all_tasks or (args.keep_best_task1_net and task == 0):
                # Save state of model
                torch.save(agent.model.state_dict(), os.path.join(get_out_path(args), "model_state_epoch_" + str(epoch) + ".pth"))

        if (args.keep_best_net_all_tasks or (args.keep_best_task1_net and task == 0)) and args.n_epoch_first_task > 1:
            # Reload state of network when it had highest test accuracy on first task
            max_acc = max(test_accs_all_epochs[0])
            max_acc_ind = test_accs_all_epochs[0].index(max_acc)
            print("Test accs on task + " + str(task) + ": " + str(test_accs_all_epochs[0]))
            print("Loading model parameters with this max test acc: " + str(max_acc))
            agent.model.load_state_dict(torch.load(
                os.path.join(get_out_path(args), "model_state_epoch_" + str(max_acc_ind) + ".pth"))
            )
            reload_test_acc, test_time = agent.validation(test_loader)
            print(' * Test Acc (after reloading best model): {acc:.3f}, Time: {time:.2f}'.format(acc=test_acc, time=test_time))
            assert reload_test_acc == max_acc, "Test accuracy of reloaded model does not match original highest test accuracy. Is the model saving and loading its state correctly?"

            # Set the test/val accs to be stored for this task to those corresponding to the best-performing network
            test_acc = max_acc
            test_acc_1st = test_accs_1st_all_epochs[0][max_acc_ind]
            if args.validate:
                val_acc = val_accs_all_epochs[0][max_acc_ind]

            # Delete saved network states
            for save_num in range(len(test_accs_all_epochs[0])):
                os.remove(os.path.join(get_out_path(args), "model_state_epoch_" + str(save_num) + ".pth"))

        # after all the epochs, store test_acc
        test_accs.append(test_acc)
        test_accs_1st.append(test_acc_1st)

        # same with val acc
        if val_data is not None:
            val_accs.append(val_acc)

    return test_accs_1st, test_accs, val_accs, test_accs_all_epochs, test_accs_1st_all_epochs


def get_args(argv):

    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='core50', help="Name of the dataset to use, e.g. 'core50', 'toybox', 'ilab2mlight'")

    # stream vs offline learning
    parser.add_argument('--offline', default = False, action = 'store_true', dest = 'offline', help = "offline vs online (stream learning) training")

    # scenario/task
    parser.add_argument('--scenario', type = str, default = 'iid', help = "How to set up tasks, e.g. iid => randomly assign data to each task")
    parser.add_argument('--n_runs', type = int, default = 1, help = "Number of times to repeat the experiment with different data orderings")

    # model hyperparameters/type
    parser.add_argument('--model_type', type=str, default='resnet', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='ResNet18', help="The name of actual model for the backbone")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--pretrained', default = False, dest = 'pretrained', action = 'store_true')
    parser.add_argument('--freeze_batchnorm', default = False, dest = 'freeze_batchnorm', action = 'store_true')
    parser.add_argument('--freeze_memory', default = False, dest = 'freeze_memory', action = 'store_true')
    parser.add_argument('--freeze_feature_extract', default = False, dest = 'freeze_feature_extract', action = 'store_true')
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--n_epoch', type = int, default = 1, help="Number of epochs to train")
    parser.add_argument('--n_epoch_first_task', type=int, default=None, help="Number of epochs to train on the first task (may be different from n_epoch, which is used for the other tasks)")
    parser.add_argument('--keep_best_task1_net', default=False, dest='keep_best_task1_net', action='store_true', help="When training for multiple epochs on task 1, retrieve the network state (among those after each epoch) with best testing accuracy for learning subsequent tasks")
    parser.add_argument('--keep_best_net_all_tasks', default=False, dest='keep_best_net_all_tasks', action='store_true', help="When training for multiple epochs on more than one task: for each task, retrieve the network state (among those after each epoch) with best testing accuracy for learning subsequent tasks")

    # keep track of validation accuracy
    parser.add_argument('--validate', default = False, action = 'store_true',  dest = 'validate', help = "To keep track of validation accuracy or not")

    # for regularization models
    parser.add_argument('--reg_coef', type=float, default=1, help="The coefficient for regularization. Larger means less plasilicity. ")

    # for replay models
    parser.add_argument('--memory_size', type=int, default=1200, help="Number of training examples to keep in memory")

    # for augmented memory model
    parser.add_argument('--memory_Nslots', type=int, default=100, help="Number of memory slots to keep in memory")
    parser.add_argument('--memory_Nfeat', type=int, default=512, help="Feature dim per memory slot to keep in memory")
    parser.add_argument('--visualize', default = False, action = 'store_true',  dest = 'visualize', help = "To visualize memory and attentions (only valid for AugMem")

    # directories
    #parser.add_argument('--dataroot', type = str, default = 'data/core50', help = "Directory that contains the data")
    parser.add_argument('--dataroot', type = str, default = '/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50', help = "Directory that contains the data")
    #parser.add_argument('--dataroot', type = str, default = '/home/mengmi/Projects/Proj_CL_NTM/data/core50', help = "Directory that contains the data")
    parser.add_argument('--filelist_root', type = str, default = 'dataloaders', help = "Directory that contains the filelists for each task")
    parser.add_argument('--output_dir', default='core50_outputs', help="Where to store accuracy table")
    parser.add_argument('--custom_folder', default=None, type=str, help="a custom subdirectory to store results")

    # gpu/cpu settings
    parser.add_argument('--gpuid', nargs="+", type=int, default=[-1],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--n_workers', default=1, type = int, help="Number of cpu workers for dataloader")

    # return parsed arguments
    args = parser.parse_args(argv)
    return args


def main():

    # get command line arguments
    args = get_args(sys.argv[1:])

    # appending path to cwd to directories
    args.dataroot = os.path.join(os.getcwd(),args.dataroot)
    args.output_dir = os.path.join(os.getcwd(),args.output_dir)

    # ensure that a valid scenario has been passed
    if args.scenario not in ['iid', 'class_iid', 'instance', 'class_instance']:
        print('Invalid scenario passed, must be one of: iid, class_iid, instance, class_instance')
        return

    # setting seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    test_accs = []
    test_accs_1st = []

    val_accs = []

    test_accs_all_epochs = []
    test_accs_1st_all_epochs = []

    # iterate over runs
    for r in range(args.n_runs):
        print('=============Stream Learning Run ' + str(r) + '=============')
        test_acc_1st, test_acc, val_acc, test_acc_all_epochs, test_acc_1st_all_epochs = run(args, r)
        test_accs.append(test_acc)
        test_accs_1st.append(test_acc_1st)
        val_accs.append(val_acc)
        test_accs_all_epochs.append(test_acc_all_epochs)
        test_accs_1st_all_epochs.append(test_acc_1st_all_epochs)

    # converting list of list of testing accuracies for each run to a dataframe
    test_df = pd.DataFrame(test_accs)
    test_df_1st = pd.DataFrame(test_accs_1st)
    val_df = pd.DataFrame(val_accs)
    test_all_epochs_dfs = [pd.DataFrame(accs) for accs in test_accs_all_epochs]
    test_1st_all_epochs_dfs = [pd.DataFrame(accs) for accs in test_accs_1st_all_epochs]

    total_path = get_out_path(args)

    # printing test acc dataframe
    print("testing accuracies")
    print(test_df)

    print("testing accuracies --- 1st task")
    print(test_df_1st)

    # printing val_acc dataframe
    print("validation accuracies")
    print(val_df)

    # writing testing accuracy to csv
    test_df_1st.to_csv(os.path.join(total_path,'test_task1.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(total_path,'test.csv'), index=False, header=False)

    # writing validation accuracy to csv, will be empty if no validation is performed
    val_df.to_csv(os.path.join(total_path,'val.csv'), index=False, header=False)

    # writing testing accuracies across all epochs to cvs
    for nrun, df in enumerate(test_all_epochs_dfs):
        df.to_csv(os.path.join(total_path, 'test_all_epochs_run' + str(nrun) + '.csv'), index=False, header=False)

    for nrun, df in enumerate(test_1st_all_epochs_dfs):
        df.to_csv(os.path.join(total_path, 'test_task1_all_epochs_run' + str(nrun) + '.csv'), index=False, header=False)

    # writing hyperparameters
    args_dict = vars(args)
    with open(os.path.join(total_path,'hyperparams.csv'), 'w') as f:
        for key, val in args_dict.items():
            f.write("{key},{val}\n".format(key=key, val=val))


if __name__ == '__main__':

    main()
