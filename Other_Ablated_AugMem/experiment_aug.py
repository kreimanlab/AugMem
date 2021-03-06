import os
import sys
import argparse
import torch
import pandas as pd
from dataloaders import datasets
from torchvision import transforms
import agents


def run(args, run):
        
    # read dataframe containing information for each task
    if args.offline:
        task_df = pd.read_csv(os.path.join('dataloaders', 'task_filelists', args.scenario, 'run' + str(run), 'stream', 'train_all.txt'), index_col = 0)
    else:
        task_df = pd.read_csv(os.path.join('dataloaders', 'task_filelists', args.scenario, 'run' + str(run), 'offline', 'train_all.txt'), index_col = 0)
    
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
        'n_class': 10,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'model_type' : args.model_type,
        'model_name' : args.model_name,
        'agent_type' : args.agent_type,
        'agent_name' : args.agent_name,
        'model_weights': args.model_weights,
        'memory_weights': args.memory_weights,
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
        'memory_topK':args.memory_topK,
        'freeze_batchnorm': args.freeze_batchnorm,
        'freeze_memory': args.freeze_memory,
        'replay_coef':args.replay_coef,
        'replay_times':args.replay_times,
        'ntask':len(tasks),
        'mem_focus_beta':args.mem_sparse,
        'mem_pretrain': args.mem_pretrain,
        'logit_coef':args.logit_coef,
        'herding_mode':args.herding_mode
        }
        
    # initialize agent
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
        
    # image transformations
    composed = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
    
    # get test data
    test_data = datasets.CORE50(
                dataroot = args.dataroot, filelist_root = args.filelist_root, scenario = args.scenario, offline = args.offline, run = run, train = False, transform=composed)
    
    if args.validate:
        # splitting test set into test and validation
        test_size = int(0.75 * len(test_data))
        val_size = len(test_data) - test_size
        test_data, val_data = torch.utils.data.random_split(test_data, [test_size, val_size])
    else:
        val_data = None

        
    test_accs = train(agent, composed, args, run, tasks, active_out_nodes, test_data, val_data)
        
    return (test_accs)
    

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
    
    # iterate over tasks
    for task in range(ntask):
        
        print('=============Training Task ' + str(task) + '=============')
    
        agent.active_out_nodes = active_out_nodes[task]
    
        print('Active output nodes for this task: ')
        print(agent.active_out_nodes)
        
        if task == 0:
            args.n_epoch = args.first_times
        else:
            args.n_epoch = 1
                
        for epoch in range(args.n_epoch):
                
            print('===' + args.agent_name + '; Epoch ' + str(epoch) + '; RUN ' + str(run) + '; TASK ' + str(task))
    
            # get training data pertaining to chosen scenario, task, run
            train_data = datasets.CORE50(
                    dataroot = args.dataroot, filelist_root = args.filelist_root, scenario = args.scenario, offline = args.offline, run = run, batch = task, transform=transforms)
            
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
            agent.learn_stream(train_loader, task)            
            #agent.save_memory('pretrain/cifar')
            
            # validate if applicable
            if args.validate:
                val_acc_out, val_acc_direct, val_time = agent.validation(val_loader)
                print(' * Val Acc: A-out {val_acc_out:.3f}, A-direct {val_acc_direct:.3f}, Time: {time:.2f}'.format(val_acc_out=val_acc_out, val_acc_direct=val_acc_direct, time=val_time))
    
            test_acc_out, test_acc_direct, test_time = agent.validation(test_loader)            
            print(' * Test Acc: A-out {test_acc_out:.3f}, A-direct {test_acc_direct:.3f}, Time: {time:.2f}'.format(test_acc_out=test_acc_out, test_acc_direct= test_acc_direct, time=test_time))
            
            test_acc_out_1st, test_acc_direct_1st, test_time_1st = agent.validation(test_loader_1st)            
            print(' * Test Acc (1st task): A-out {test_acc_out:.3f}, A-direct {test_acc_direct:.3f}, Time: {time:.2f}'.format(test_acc_out=test_acc_out_1st, test_acc_direct= test_acc_direct_1st, time=test_time_1st))
            
            if args.visualize:
                attread_filename = 'visualization/' + args.scenario + '/' + args.scenario + '_run_' + str(run) + '_task_' + str(task) + '_epoch_' + str(epoch)
                agent.visualize_att_read(attread_filename)
                agent.visualize_memory(attread_filename)
            
        # after all the epochs, store test_acc
        test_accs.append(test_acc_direct)  
        test_accs_1st.append(test_acc_direct_1st)
        
        # same with val acc
        if val_data is not None:
            val_accs.append(val_acc_out)
    
    return (test_accs_1st, test_accs, val_accs)


def get_args(argv):
    
    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()
    
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
    parser.add_argument('--memory_weights', type=str, default=None,
                        help="The path to the file for the memory weights (*.pth).")
    parser.add_argument('--n_epoch', type = int, default = 1, help="Number of epochs to train")
    
    # keep track of validation accuracy
    parser.add_argument('--validate', default = False, action = 'store_true',  dest = 'validate', help = "To keep track of validation accuracy or not")
    
    # for regularization models
    parser.add_argument('--reg_coef', type=float, default=1, help="The coefficient for regularization. Larger means less plasilicity. ")
    parser.add_argument('--logit_coef', type=float, default=1, help="The coefficient for regularization. Larger means less plasilicity. ")
    
    # for replay models
    parser.add_argument('--memory_size', type=int, default=1200, help="Number of training examples to keep in memory")
    parser.add_argument('--replay_coef', type=float, default=1, help="The coefficient for replays. Larger means less plasilicity. ")
    parser.add_argument('--replay_times', type=int, default=1, help="The number of times to replay per batch. ")
    parser.add_argument('--mem_sparse', type=float, default=5, help="The number of times to replay per batch. ")
    parser.add_argument('--first_times', type=int, default=1, help="The number of times to go through task 0. ")
    parser.add_argument('--mem_pretrain', default = False, action = 'store_true',  dest = 'mem_pretrain', help = "To load pre-trained mem weights from kmeans on cifar100")
    
    parser.add_argument('--herding_mode', default = False, action = 'store_true',  dest = 'herding_mode', help = "To use herding to select examplar")
    
    # for augmented memory model
    parser.add_argument('--memory_topK', type=int, default=1, help="Number of memory slots to keep in memory")
    parser.add_argument('--memory_Nslots', type=int, default=100, help="Number of memory slots to keep in memory")
    parser.add_argument('--memory_Nfeat', type=int, default=512, help="Feature dim per memory slot to keep in memory")
    parser.add_argument('--visualize', default = False, action = 'store_true',  dest = 'visualize', help = "To visualize memory and attentions (only valid for AugMem")
    
    # directories
    #parser.add_argument('--dataroot', type = str, default = 'data/core50', help = "Directory that contains the data")    
    parser.add_argument('--dataroot', type = str, default = '/media/data/Datasets/Core50', help = "Directory that contains the data")
    #parser.add_argument('--dataroot', type = str, default = '/home/mengmi/Projects/Proj_CL_NTM/data/core50', help = "Directory that contains the data")
    parser.add_argument('--filelist_root', type = str, default = 'dataloaders', help = "Directory that contains the filelists for each task")
    parser.add_argument('--output_dir', default='outputs',
                        help="Where to store accuracy table")
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
            
    test_accs = []
    test_accs_1st = []
    val_accs = []
    
    # iterate over runs
    for r in range(args.n_runs):
        print('=============Stream Learning Run ' + str(r) + '=============')
        test_acc_1st, test_acc, val_acc = run(args, r)
        test_accs.append(test_acc)
        test_accs_1st.append(test_acc_1st)
        val_accs.append(val_acc)
    
        # converting list of list of testing accuracies for each run to a dataframe
        test_df = pd.DataFrame(test_accs)
        test_df_1st = pd.DataFrame(test_accs_1st)
        val_df = pd.DataFrame(val_accs)
        
        if args.custom_folder is None:
            if args.offline:
                subdir = args.agent_name + '_' + args.model_name + '_' + 'offline/' 
            else:
                subdir = args.agent_name + '_' + args.model_name
        else:
            subdir = args.custom_folder
                            
        outdir = os.path.join(args.output_dir, args.scenario)
        
        total_path = os.path.join(outdir, subdir)
        
        # make output directory if it doesn't already exist
        if not os.path.exists(total_path):
            os.makedirs(total_path)
            
        # printing test acc dataframe
        print("testing accuracies")
        print(test_df)
        
        print("testing accuracies --- 1st task")
        print(test_df_1st)
        
        # printing val_acc dataframe
        print("validation accuracies")
        print(val_df)
        
        # writing testing accuracy to csv
        print(total_path)
        test_df_1st.to_csv(os.path.join(total_path,'test_task1.csv'), index = False, header = False)
        test_df.to_csv(os.path.join(total_path,'test.csv'), index = False, header = False)
        
        # writing validation accuracy to csv, will be empty if no validation is performed
        val_df.to_csv(os.path.join(total_path,'val.csv'), index = False, header = False)
        
        # writing hyperparameters
        args_dict = vars(args)
        with open(os.path.join(total_path,'hyperparams.csv'), 'w') as f:
            for key, val in args_dict.items():
                f.write("{key},{val}\n".format(key=key, val=val))

if __name__ == '__main__':
    
    main()
    
    
    