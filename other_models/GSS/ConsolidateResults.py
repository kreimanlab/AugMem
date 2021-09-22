import pandas as pd
import os
import csv

numrun = 10
paradigm = 'class_instance'

test_accs_all = []
for run in range(numrun):
    filename = 'output/core50/test_' + paradigm + '_run_' + str(run) + '.csv'
    rows = []
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)     
      
        # extracting each data row one by one
        for row in csvreader:
            rows.append(float(row[1]))
        
        print(rows)
        test_accs_all.append(rows) 


test_df = pd.DataFrame(test_accs_all)
print("testing accuracies")
print(test_df)
test_df.to_csv(os.path.join('output/test_' + paradigm + '.csv'), index = False, header = False)