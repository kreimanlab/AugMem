class toybox():
	def __init__(self, paradigm, run):
		self.batch_num = 6
		self.rootdir = '/home/rushikesh/code/dataloaders/toybox_task_filelists/'
		
		self.train_data = []
		self.train_labels = []
		self.train_groups = [[],[],[],[],[],[]]
		for b in range(self.batch_num):
			with open( self.rootdir + paradigm + '/run' + str(run) + '/stream/train_task_' + str(b).zfill(2) + '_filelist.txt','r') as f:
				for i, line in enumerate(f):
					if line.strip():
						path, label = line.split()
						
						self.train_groups[b].append((path, int(label)))
						self.train_data.append(path)
						self.train_labels.append(int(label))
		self.train = {'data': self.train_data,'fine_labels': self.train_labels}
		self.val_groups = self.train_groups.copy()        
			   
		self.test_data = []
		self.test_labels = []
		self.test_groups = [[],[],[],[],[],[]]
		groupsid = {'0':0,'1':0,
					'2':1,'3':1,
					'4':2,'5':2,
					'6':3,'7':3,
					'8':4,'9':4,
					'10':5,'11':5}
		with open( self.rootdir + paradigm + '/run' + str(run) + '/stream/test_filelist.txt','r') as f:
				for i, line in enumerate(f):
					if line.strip():
						path, label = line.split()
						gp = groupsid[label]
						#print("label, gp")
						#print(label, gp)
						self.test_groups[gp].append((path, int(label)))
						self.test_data.append(path)
						self.test_labels.append(int(label))
					
		self.test = {'data': self.test_data,'fine_labels': self.test_labels}
	def getNextClasses(self, test_id):

		return self.train_groups[test_id], self.val_groups[test_id], self.test_groups[test_id]   #self.test_grps #
