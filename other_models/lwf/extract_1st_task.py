inFile = open("15Nov_ilab_01.txt")
outFile = open("results_together_15Nov_ilab_01.txt", "w")
buffer = []
keepCurrentSet = False
for line in inFile:
    #buffer.append(line)
    if line.startswith("Accuracy for task 1 is"):
        #---- starts a new data set
        keepCurrentSet = True
        continue
        '''
        if keepCurrentSet:
            outFile.write(",".join(line))
            outFile.write("\n")
            keepCurrentSet = False
        '''
        #now reset our state
        #keepCurrentSet = True
        #buffer = []
    #elif line.startswith("extractme"):
        #keepCurrentSet = True
    if keepCurrentSet:
        outFile.write(line)
        #outFile.write("\n")
        keepCurrentSet = False

inFile.close()
outFile.close()