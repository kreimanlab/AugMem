import sys
import os
import pandas as pd

def main():
    
    # get command line arguments
    args = sys.argv[1:]
    fname = args[0]
    wait_int = int(args[1])
    
    # reading from csv
    name = './gridsearches/' + fname + '.csv'
    grid = pd.read_csv(name, index_col=0)
    
    script = open('./gridsearches/' + fname + '.sh', 'w')
    
    # parsing gridsearch and running experiments
    for index, row in grid.iterrows():
        # default command
        command = 'python -u experiment.py'
        d = row.to_dict()
        for k,v in d.items():
            # if true boolean, pass flag
            if v is True:
                command += ' --' + k
            # if false, do nothing
            elif v is False:
                pass
            # otherwise, append argument after flag
            else:
                command += ' --' + k + ' ' + str(v)
        # ensuring validation is on, since we're doing a grid search
        command += ' --validate'
        # get unique number of values in each column
        unique_per_col = grid.nunique()
        # get columns where more than one unique values exist
        # these are the parameters that are being gridsearched (with the exception of gpuid)
        search_cols = unique_per_col[unique_per_col > 1].index
        # filtering the parameters based on the ones that are being searched
        run_identifier = {k:v for k,v in d.items() if k in search_cols}
        # removing gpuid
        if 'gpuid' in run_identifier.keys():
            del run_identifier['gpuid']
        # removing scenario as this information is already included in directory structure
        del run_identifier['scenario']
        # converting to string and removing colons & apostrophes & brackets
        run_identifier = str(run_identifier).replace(':', '-').replace("'", '').replace('{', '').replace('}','').replace(' ', '')
        # based on the run identifier, define the subdirectory where the results will be stored
        subdir = fname + '/' + run_identifier + '/'
        # pass this directory to the experiment
        command += ' --custom_folder ' + subdir  
        # get the output of the file and store in log file in proper directory
        total_path = './toybox_gridsearch_outputs/' + d['scenario'] + '/' + subdir 
        # making directory if it does not exist already
        if not os.path.exists(total_path):
            os.makedirs(total_path)
        log_storage = total_path + 'log.log'
        command += ' | tee ' + log_storage # + ' &'
        print("Writing this command to " + fname + '.sh')
        print(command)
        print('-------------')
        #os.system(command)
        if index % wait_int == 0 and index != 0:
            script.write('wait \n')
        script.write(command + '\n')
        

if __name__ == '__main__':
    
    main()
