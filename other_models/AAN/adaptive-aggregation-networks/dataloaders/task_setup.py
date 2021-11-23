import pandas as pd
import numpy as np
import os
import sys
import math
import argparse

#core50 dataset components
#10 classes; 
#5 objects per class; 
#50 objects in total
#11 sessions per object (8 sessions for train, 3 for test)
#instance is defined as (object, session)

#Toybox dataset components
#12 classes
#30 objects per class
#360 objects in total
#10 sessions per object (12 in original dataset, "present" and "absent" transformations discarded)
TOYBOX_FRAMES_PER_SESSION = 15

#iLab-2M-Light dataset components
# 14 classes ("semi" removed from original 15)
# 28 objects per class
# 392 objects in total
# 8 sessions per object
ILAB2MLIGHT_FRAMES_PER_SESSION = 15


# sample every x frames from core50 dataset
def sample_frames_core50(examples, sample_rate=20, first_frame=10, max_frames=None):

    # for some reason, some image frames are missing before the 10th frame, so we start sampling at the 10th frame
    sampled = examples[examples.im_num.isin(np.arange(first_frame,300,sample_rate))]
    return sampled


# split into training and testing based on choice of session    
def train_test_split(dataset, outdir, test_sess):

    # get directory map of dataset (core50, toybox, etc)
    dirmap = pd.read_csv(os.path.join(outdir, dataset + '_dirmap.csv'))

    # sample frames from sessions
    # core50 videos were taken 20fps, so sampling every 20 images is equivalent to sampling one frame per second
    if dataset == 'core50':
        examples = sample_frames_core50(dirmap)
    elif dataset == 'toybox':
        # toybox pre-sampled in toybox_sample.py due to the complexity of this dataset's structure
        examples = dirmap
    elif dataset == 'ilab2mlight':
        # ilab2mlight pre-sampled in ilab2mlight_sample.py due to the complexity of this dataset's structure
        examples = dirmap
    elif dataset == 'cifar100':
        examples = dirmap
    else:
        raise ValueError("Invalid dataset name")
    
    # split based on which sessions are designated train and test sessions
    train = examples[~examples.session.isin(test_sess)].reset_index(drop=True)
    test = examples[examples.session.isin(test_sess)].reset_index(drop=True)
    
    return train, test
    

# setup files containing image paths/labels for each task in the iid scenario
def iid_task_setup(dataset='core50', n_runs = 10, task_size = 1200, sample_rate = 20, test_sess = [3,7,10], offline = False, outdir = 'dataloaders'):
    
    # split into training and testing
    # default is to use sessions 3,7, & 10 for testing, the rest for training
    train, test = train_test_split(dataset, outdir, test_sess)
    
    # creating tasks for each run
    for run in range(n_runs):
        # passing a copy so the original train df is not modified
        iid_task_filelist(train.copy(), test, run, task_size, offline, outdir, dataset=dataset)
    

# setup files containing image paths/labels for each task in the class iid scenario
# nclass is number of classes per task
def class_iid_task_setup(dataset='core50', n_runs = 10, n_class = 2, sample_rate = 20, test_sess = [3,7,10], offline = False, outdir='dataloaders'):
    
    # split into training and testing
    # default is to use sessions 3,7, & 10 for testing, the rest for training
    train, test = train_test_split(dataset, outdir, test_sess)
    
    # creating tasks for each run
    for run in range(n_runs):
        # passing a copy so the original train df is not modified
        class_iid_task_filelist(train.copy(), test, run, n_class, offline, outdir, dataset=dataset)


# setup files containing image paths/labels for each task in the instance setting
# ninstance is number of instances per task
def instance_task_setup(dataset='core50', n_runs = 10, n_instance = 80, sample_rate = 20, test_sess = [3,7,10], offline = False, outdir = 'dataloaders'):

    # split into training and testing
    # default is to use sessions 3,7, & 10 for testing, the rest for training
    train, test = train_test_split(dataset, outdir, test_sess)
    
    # creating tasks for each run
    for run in range(n_runs):
        # passing a copy so the original train df is not modified
        instance_task_filelist(train.copy(), test, run, n_instance, sample_rate, offline, outdir, dataset=dataset)
        

# setup files containing image paths/labels for each task in the class scenario
# nclass is number of classes per task
def class_instance_task_setup(dataset='core50', n_runs = 10, n_class = 2, sample_rate = 20, test_sess = [3,7,10], offline = False, outdir = 'dataloaders'):
    
    # split into training and testing
    # default is to use sessions 3,7, & 10 for testing, the rest for training
    train, test = train_test_split(dataset, outdir, test_sess)
    
    # creating tasks for each run
    for run in range(n_runs):
        # passing a copy so the original train df is not modified
        class_instance_task_filelist(train.copy(), test, run, n_class, offline,  outdir, dataset=dataset)


# setup tasks where each task is iid (independently and identically distributed)
def iid_task_filelist(train, test, run, task_size, offline, outdir, dataset='core50'):
    
    # defining directory to dump file
    if offline:
        run_dir = outdir + '/' + dataset + '_task_filelists/iid/run' + str(run) + '/offline/'
    else:
        run_dir = outdir + '/' + dataset + '_task_filelists/iid/run' + str(run) + '/stream/'
    
    # creating directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir) 
     
    # initializing 
    train_final = pd.DataFrame(columns = train.columns)
    
    # shuffling training data, using a different seed for each run
    train_shuffled = train.sample(frac = 1, random_state = run).reset_index(drop = True)
    
    # last task may have fewer examples, if no. examples and task size do not divide properly
    n_task = math.ceil(train.shape[0] / task_size)
    
    # assigning examples to each task
    for b in range(n_task):
        # if offline, we want all previous examples as well
        # also want to shuffle them
        if offline:
            train_task = train_shuffled.loc[: (b+1) * task_size].sample(frac = 1, random_state = run)
        else:
            train_task = train_shuffled.loc[b * task_size: (b+1) * task_size]
        
        # assigning task
        train_task['task'] = b
        
        # appending task to train_final
        train_final = train_final.append(train_task)    
     
    # defining labels based on order of appearance of classes
    classes = train_final['class'].unique()
    class_to_label = {classes[i]:i for i in range(len(classes))}
    train_final['label'] = train_final['class'].map(class_to_label)
    test['label'] = test['class'].map(class_to_label)


    # writing tasks to .txt files
    for b in range(n_task):
        train_final[train_final.task == b][['im_path', 'label']].to_csv(run_dir + '/train_task_' + str(b).zfill(2) + '_filelist.txt', sep = ' ', index = False, header = False)
    
    # write test set to .txt as well
    test[['im_path', 'label']].to_csv(run_dir + '/test_filelist.txt', sep = ' ', index = False, header = False)
    
    # writing the train df to .txt
    train_final.to_csv(run_dir + '/train_all.txt')

        
def class_iid_task_filelist(train, test, run, n_class, offline, outdir, dataset='core50'):
    
    # defining directory to dump file
    if offline:
        run_dir = outdir + '/' + dataset + '_task_filelists/class_iid/run' + str(run) + '/offline/'
    else:
        run_dir = outdir + '/' + dataset + '_task_filelists/class_iid/run' + str(run) + '/stream/'
    
    # creating directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # setting the random seed
    np.random.seed(run)
    
    # shuffling the list of classes
    shuffled_classes = train['class'].unique()
    np.random.shuffle(shuffled_classes)
    
    # number of tasks
    n_task = math.ceil(len(shuffled_classes) / n_class)
    
    # initializing task column
    train['task'] = -1
    
    # initializing 
    train_final = pd.DataFrame(columns = train.columns)
    
    # iterating over classes and assigning examples for each class a task
    for b in range(n_task):
        
        # assigning classes to task
        # shuffline again if offline
        if offline:
            task_classes = shuffled_classes[:(b+1)*n_class]
            np.random.shuffle(task_classes)
        else:  
            task_classes = shuffled_classes[(b*n_class):(b+1)*n_class]
        
        # shuffling task and assigning task number
        train_task = train.loc[train['class'].isin(task_classes)].sample(frac = 1, random_state = run)
        train_task['task'] = b
        
        # appending task to train_final
        train_final = train_final.append(train_task)
        
    # defining labels based on order of appearance of classes
    classes = train_final['class'].unique()
    class_to_label = {classes[i]:i for i in range(len(classes))}
    train_final['label'] = train_final['class'].map(class_to_label)
    test['label'] = test['class'].map(class_to_label)
    
    for b in range(n_task):
        
        # writing task filelist
        train_final[train_final.task == b][['im_path', 'label']].to_csv(run_dir + '/train_task_' + str(b).zfill(2) + '_filelist.txt', sep = ' ', index = False, header = False)

            
    # write test set to .txt as well
    test[['im_path', 'label']].to_csv(run_dir + '/test_filelist.txt', sep = ' ', index = False, header = False)
    
    # writing the train df to .txt
    train_final.reset_index(drop = True, inplace = True)
    train_final.to_csv(run_dir + '/train_all.txt')

    
def instance_task_filelist(train, test, run, n_instance, sample_rate, offline, outdir, dataset='core50'):
    
    # defining directory to dump file
    if offline:
        run_dir = outdir + '/' + dataset + '_task_filelists/instance/run' + str(run) + '/offline/'
    else:
        run_dir = outdir + '/' + dataset + '_task_filelists/instance/run' + str(run) + '/stream/'
    
    # creating directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # setting random seed
    np.random.seed(run)
    
    # getting unique instances of objects
    instances = np.unique(train[['object', 'session']].values, axis = 0)
    
    # shuffling
    np.random.shuffle(instances)
    
    # calculating the number of training examples per task
    # MT: n_instance is actually the number of SESSIONS per task
    if dataset == 'core50':
        n_ex = n_instance * (300 / sample_rate)
    elif dataset == 'toybox':
        n_ex = n_instance * TOYBOX_FRAMES_PER_SESSION
    elif dataset == 'ilab2mlight':
        n_ex = n_instance * ILAB2MLIGHT_FRAMES_PER_SESSION
    else:
        raise ValueError("Invalid dataset name, use 'core50', 'toybox', or 'ilab2mlight'")
    
    # last task might have fewer instances due to no. train examples not divinding evenly
    # MT: train.shape[0] is the number of images in the train set
    n_task = math.ceil(train.shape[0] / n_ex)
    
    # initializing task column
    train['task'] = -1
    
    # initializing 
    train_final = pd.DataFrame(columns = train.columns)
    
    # iterate over tasks
    for b in range(n_task):
        
        # initializing
        train_task = pd.DataFrame(columns = train.columns)
        
        # assigning instances to task
        # if offline, getting all seen instances
        # also shuffline if offline
        if offline:
            task_instances = instances[:(b+1)*n_instance]
            np.random.shuffle(task_instances)
        else:  
            task_instances = instances[(b*n_instance):(b+1)*n_instance]
        
        # iterate over instances assigned to that task
        for inst in task_instances:
                    
            # get examples that match that instance, ensure its sorted temporally
            inst_task = train[(train.object == inst[0]) & (train.session == inst[1])].sort_values('im_num')
            
            # assign task
            inst_task['task'] = b
            
            # append to examples for that task
            train_task = train_task.append(inst_task)
            
        # append to train_final
        train_final = train_final.append(train_task)
    
    # defining labels based on order of appearance of classes
    classes = train_final['class'].unique()
    class_to_label = {classes[i]:i for i in range(len(classes))}
    train_final['label'] = train_final['class'].map(class_to_label)
    test['label'] = test['class'].map(class_to_label)
    
    for b in range(n_task):
            
        # write task to file
        train_final[train_final.task == b][['im_path', 'label']].to_csv(run_dir + '/train_task_' + str(b).zfill(2) + '_filelist.txt', sep = ' ', index = False, header = False)

        
    # write test set to .txt as well
    test[['im_path', 'label']].to_csv(run_dir + '/test_filelist.txt', sep = ' ', index = False, header = False)
    
    # writing the train df to .txt
    train_final.reset_index(drop = True, inplace = True)
    train_final.to_csv(run_dir + '/train_all.txt')      
        
    
def class_instance_task_filelist(train, test, run, n_class, offline, outdir, dataset='core50'):
    
    # defining directory to dump file
    if offline:
        run_dir = outdir + '/' + dataset + '_task_filelists/class_instance/run' + str(run) + '/offline/'
    else:
        run_dir = outdir + '/' + dataset + '_task_filelists/class_instance/run' + str(run) + '/stream/'
    
    # creating directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # setting the random seed
    np.random.seed(run)
    
    # shuffling the list of classes
    shuffled_classes = train['class'].unique()
    np.random.shuffle(shuffled_classes)
    
    # number of tasks
    n_task = math.ceil(len(shuffled_classes) / n_class)
    
    # initializing task column
    train['task'] = -1
    
    # initializing 
    train_final = pd.DataFrame(columns = train.columns)
    
    # iterating over classes and assigning examples for each class a task
    for b in range(n_task):
        
        train_task = pd.DataFrame(columns = train.columns)
        
        # assigning classes to task
        # if offline, getting all previously seen classes and shuffline
        if offline:
            task_classes = shuffled_classes[:(b+1)*n_class]
            np.random.shuffle(task_classes)
        else:  
            task_classes = shuffled_classes[(b*n_class):(b+1)*n_class]
                    
        # get examples pertaining to these classes
        train_class = train[train['class'].isin(task_classes)]
        
        class_instances = np.unique(train_class[['object', 'session']].values, axis = 0)
        np.random.shuffle(class_instances)
        
        for inst in class_instances:
                        
            # get examples that match that instance, ensure its sorted temporally
            inst_task = train_class[(train_class.object == inst[0]) & (train_class.session == inst[1])].sort_values('im_num')
            
            # assign task
            inst_task['task'] = b
            
            # append to examples for that task
            train_task = train_task.append(inst_task)
        
        # appending task
        train_final = train_final.append(train_task)
        
    # defining labels based on order of appearance of classes
    classes = train_final['class'].unique()
    class_to_label = {classes[i]:i for i in range(len(classes))}
    train_final['label'] = train_final['class'].map(class_to_label)
    test['label'] = test['class'].map(class_to_label)
    
    for b in range(n_task):
        # writing task filelist
        train_final[train_final.task == b][['im_path', 'label']].to_csv(run_dir + '/train_task_' + str(b).zfill(2) + '_filelist.txt', sep = ' ', index = False, header = False)
        
            
    # write test set to .txt as well
    test[['im_path', 'label']].to_csv(run_dir + '/test_filelist.txt', sep = ' ', index = False, header = False)
    
    # writing the train df to csv
    train_final.reset_index(drop = True, inplace = True)
    train_final.to_csv(run_dir + '/train_all.txt')


def write_task_filelists(args):
    
    if args.scenario == 'iid':
        iid_task_setup(dataset=args.dataset, n_runs = args.n_runs, task_size = args.task_size_iid, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root)
        
    elif args.scenario == 'class_iid':
        class_iid_task_setup(dataset=args.dataset, n_runs = args.n_runs, n_class = args.n_class, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root)
    
    elif args.scenario == 'instance':
        instance_task_setup(dataset=args.dataset, n_runs = args.n_runs, n_instance = args.n_instance, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root)
        
    elif args.scenario == 'class_instance':
        class_instance_task_setup(dataset=args.dataset, n_runs = args.n_runs, n_class = args.n_class, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root)


def get_args(argv):
    
    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()

    # scenario/task
    parser.add_argument('--dataset', type=str, default='core50', help="Name of the dataset to use, e.g. 'core50', 'toybox', 'ilab2mlight'")
    parser.add_argument('--scenario', type=str, default='iid', help="How to set up tasks, e.g. iid => randomly assign data to each task")
    parser.add_argument('--n_runs', type=int, default=10, help="Number of times to repeat the experiment with different data orderings")
    parser.add_argument('--task_size_iid', type=int, default=1200, help="Number of examples per task, only applicable for the iid setting")
    parser.add_argument('--n_class', type=int, default=2, help="Number of classes per task, valid for the class_iid and class settings")
    parser.add_argument('--n_instance', type=int, default=80, help="Number of instances to assign to each class, valid for the instance setting") #8 sessions per obj; 5 objs; 2 classes per incremental task; thus 80 instances (note - actually SESSIONS) per task
    parser.add_argument('--sample_rate', type=int, default=20, help="Sample an image every x frames")
    parser.add_argument('--test_sess', nargs="+", default=[3, 7, 10], type=int, help="Which sessions to use for testing")
    parser.add_argument('--offline', default=False, action='store_true', dest='offline')

    # directories
    parser.add_argument('--root', type=str, default='dataloaders', help="Directory that contains the data")
    
    # return parsed arguments
    args = parser.parse_args(argv)
    return args


def main():
    
    # get command line arguments
    args = get_args(sys.argv[1:])
    
    # appending path to cwd to root
    args.root = os.path.join(os.getcwd(), args.root)
    
    # ensure that a valid scenario has been passed
    if args.scenario not in ['iid', 'class_iid', 'instance', 'class_instance']:
        print('Invalid scenario passed, must be one of: iid, class_iid, instance, class')
        return
    
    write_task_filelists(args)
    

if __name__ == '__main__':
    
    main()
    


    

