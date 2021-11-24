class cifar100():
	def __init__(self, paradigm, run):
		self.batch_num = 20
		self.rootdir = '/home/rushikesh/code/dataloaders/cifar100_task_filelists/'
		self.rain_train = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.rain_test = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.train_data = []
		self.train_labels = []
		self.train_groups = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		for b in range(self.batch_num):
			with open( self.rootdir + paradigm + '/run' + str(run) + '/stream/train_task_' + str(b).zfill(2) + '_filelist.txt','r') as f:
				#print("opened successfully")
				for i, line in enumerate(f):
					if line.strip():
						path, label = line.split()
						
						self.train_groups[b].append((path, int(label)))
						self.train_data.append(path)
						self.train_labels.append(int(label))
						self.rain_train[b].append({'klass': str(label), 'file_name': path, 'label': int(label)})
		self.train = {'data': self.train_data,'fine_labels': self.train_labels}
		self.val_groups = self.train_groups.copy()        
			   
		self.test_data = []
		self.test_labels = []
		self.test_groups = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		idss = [i for i in range(100)]

		groupsid = {'0':0,'1':0,'2':0,'3':0,'4':0,
		'5':1,'6':1,'7':1,'8':1,'9':1,
		'10':2,'11':2,'12':2,'13':2,'14':2,
		'15':3,'16':3,'17':3,'18':3,'19':3,
		'20':4,'21':4,'22':4,'23':4,'24':4,
		'25':5,'26':5,'27':5,'28':5,'29':5,
		'30':6,'31':6,'32':6,'33':6,'34':6,
		'35':7,'36':7,'37':7,'38':7,'39':7,
		'40':8,'41':8,'42':8,'43':8,'44':8,
		'45':9,'46':9,'47':9,'48':9,'49':9,
		'50':10,'51':10,'52':10,'53':10,'54':10,
		'55':11,'56':11,'57':11,'58':11,'59':11,
		'60':12,'61':12,'62':12,'63':12,'64':12,
		'65':13,'66':13,'67':13,'68':13,'69':13,
		'70':14,'71':14,'72':14,'73':14,'74':14,
		'75':15,'76':15,'77':15,'78':15,'79':15,
		'80':16,'81':16,'82':16,'83':16,'84':16,
		'85':17,'86':17,'87':17,'88':17,'89':17,
		'90':18,'91':18,'92':18,'93':18,'94':18,
		'95':19,'96':19,'97':19,'98':19,'99':19	}
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
						self.rain_test[b].append({'klass': str(label), 'file_name': path, 'label': int(label)})
					
		self.test = {'data': self.test_data,'fine_labels': self.test_labels}
		return self.rain_train,self.rain_test
	def getNextClasses(self, test_id):

		return self.train_groups[test_id], self.val_groups[test_id], self.test_groups[test_id]   #self.test_grps #
