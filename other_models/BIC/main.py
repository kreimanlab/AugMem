import torch
import numpy as np
from trainer import Trainer
import sys
from utils import *
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--dataset', default = 'toybox', type = str, help='core50 | toybox | ilab ')
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--epoch', default = 15, type = int)
parser.add_argument('--lr', default = 0.001, type = float)
parser.add_argument('--max_size', default = 20, type = int)
parser.add_argument('--total_cls', default = 10, type = int)
parser.add_argument('--numrun', default = 10, type = int)
parser.add_argument('--paradigm', default = 'class_iid', type = str)
args = parser.parse_args()


if __name__ == "__main__":
    #showGod()
    
    test_accs_all = []
    for run in range(args.numrun):
    
        trainer = Trainer(args.total_cls, args.paradigm, run,args.dataset)
        test_accs = trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
        
        test_accs_all.append(test_accs)
        
    # writing testing accuracy to csv
    #test_df_1st.to_csv(os.path.join(total_path,'test_task1.csv'), index = False, header = False)    
    test_df = pd.DataFrame(test_accs_all)
    print("testing accuracies")
    print(test_df)
    test_df.to_csv(os.path.join('output/test_' + args.paradigm + '.csv'), index = False, header = False)
