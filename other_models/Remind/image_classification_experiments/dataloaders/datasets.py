import os
import os.path
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision.datasets.folder import pil_loader


class CORE50(data.Dataset):
    """`CORE50 <https://vlomonaco.github.io/core50/>`_ Dataset, specifically
        designed for Continuous Learning and Robotic Vision applications.
        For more information and additional materials visit the official
        website `CORE50 <https://vlomonaco.github.io/core50/>`
    Args:
        root (string): Root directory of the dataset where the ``CORe50``
            dataset exists or should be downloaded.
        check_integrity (bool, optional): If True check the integrity of the
            Dataset before trying to load it.
        scenario (string, optional): Which benchmark scenario is to be tested, 'NI_inc', 'NC_inc', 'iid', etc.
        train (bool, optional): If True, creates the dataset from the training
            set, otherwise creates from test set.
        img_size (string, optional): One of the two img sizes available among
            ``128x128`` or ``350x350``.
        run (int, optional): One of the 10 runs (from 0 to 9) in which the
            training batch order is changed. Multiple runs are available because performance can vary
            greatly with ordering of classes/data
        batch (int, optional): Which one of the training incremental batches from 0 to
            max-batch - 1 to be loaded. Remember that for the ``ni``, ``nc`` and ``nic`` we
            have respectively 8, 9 and 79 incremental batches. If
            ``train=False`` this parameter will be ignored. 
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.ToTensor()``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
    Example:
        .. code:: python
            training_data = datasets.CORE50(
                '~/data/core50', transform=transforms.ToTensor(), download=True
            )
            training_loader = torch.utils.data.DataLoader(
                training_data, batch_size=128, shuffle=True, num_workers=4
            )
            test_data = datasets.CORE50(
                '~/data/core50', transform=transforms.ToTensor(), train=False,
                download=True
            )
            test_loader = torch.utils.data.DataLoader(
                training_data, batch_size=128, shuffle=True, num_workers=4
            )
            for batch in training_loader:
                imgs, labels = batch
                ...
        This is the simplest way of using the Dataset with the common Train/Test
        split. If you want to use the benchmark as in the original CORe50 paper
        (that is for continuous learning) you need to play with the parameters
        ``scenario``, ``cumul``, ``run`` and ``batch`` hence creating a number
        of Dataset objects (one for each incremental training batch and one for
        the test set).
    """
    
    filenames = {
        '128x128': 'core50_128x128.zip',
        '350x350': 'core50_350x350.zip',
        'filelists': 'task_filelists.zip'
    }

    def __init__(self, dataroot, filelist_root = 'dataloaders', scenario='iid', offline = False, train=True,
                 img_size='128x128', run=0, batch=0, transform=None, returnIndex=False,
                 target_transform=None):

        self.dataroot = os.path.expanduser(dataroot)
        self.filelist_root = os.path.expanduser(filelist_root)
        self.img_size = img_size
        self.scenario = scenario
        self.run = run
        self.batch = batch
        self.transform = transform
        self.target_transform = target_transform

        # To be filled
        self.fpath = None
        self.img_paths = []
        self.labels = []
        self.returnIndex = returnIndex

        if train:
            if offline:
                self.fpath = os.path.join(
                    self.scenario, 'run' + str(run), 'offline',
                    'train_task_' + str(batch).zfill(2) + '_filelist.txt'
                )
            else:
                self.fpath = os.path.join(
                    self.scenario, 'run' + str(run), 'stream',
                    'train_task_' + str(batch).zfill(2) + '_filelist.txt'
                    )
        else:
            # it's the last one, hence the test batch
            if offline:
                self.fpath = os.path.join(
                self.scenario, 'run' + str(run), 'offline',
                'test_filelist.txt'
                )
                
            else:
                self.fpath = os.path.join(
                    self.scenario, 'run' + str(run), 'stream',
                    'test_filelist.txt'
                )

        # Loading the filelist
        # [:-4] is to remove the .zip extension from the filename
        # path = os.path.join(self.root, self.filenames['filelists'][:-4], self.fpath)
        path = os.path.join(self.filelist_root, 'core50_task_filelists', self.fpath)
        
        # loading all the labels and image paths
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    path, label = line.split()
                    self.labels.append(int(label))
                    self.img_paths.append(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        fpath = self.img_paths[index]
        target = self.labels[index]
        
        img = pil_loader(
            os.path.join(self.dataroot, self.filenames[self.img_size][:-4], fpath)
        )

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.returnIndex:
            return img, target, index
        else:
            return img, target

    def __len__(self):

        return len(self.labels)


class Generic_Dataset(data.Dataset):

    def __init__(self, dataroot, dataset="toybox", filelist_root='dataloaders', scenario='iid', offline=False, train=True,
                 run=0, batch=0, transform=None, returnIndex=False,
                 target_transform=None):

        self.dataroot = os.path.expanduser(dataroot)
        self.filelist_root = os.path.expanduser(filelist_root)
        self.scenario = scenario
        self.run = run
        self.batch = batch
        self.transform = transform
        self.target_transform = target_transform
        self.returnIndex = returnIndex

        # To be filled
        self.fpath = None
        self.img_paths = []
        self.labels = []

        if train:
            if offline:
                self.fpath = os.path.join(
                    self.scenario, 'run' + str(run), 'offline',
                                   'train_task_' + str(batch).zfill(2) + '_filelist.txt'
                )
            else:
                self.fpath = os.path.join(
                    self.scenario, 'run' + str(run), 'stream',
                                   'train_task_' + str(batch).zfill(2) + '_filelist.txt'
                )
        else:
            # it's the last one, hence the test batch
            if offline:
                self.fpath = os.path.join(
                    self.scenario, 'run' + str(run), 'offline',
                    'test_filelist.txt'
                )

            else:
                self.fpath = os.path.join(
                    self.scenario, 'run' + str(run), 'stream',
                    'test_filelist.txt'
                )

        # Loading the filelist
        # [:-4] is to remove the .zip extension from the filename
        # path = os.path.join(self.root, self.filenames['filelists'][:-4], self.fpath)
        path = os.path.join(self.filelist_root, dataset + '_task_filelists', self.fpath)

        # loading all the labels and image paths
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    path, label = line.split()
                    self.labels.append(int(label))
                    self.img_paths.append(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        fpath = self.img_paths[index]
        target = self.labels[index]

        img = pil_loader(
            os.path.join(self.dataroot, fpath)
        )

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.returnIndex:
            return img, target, index
        else:
            return img, target

    def __len__(self):

        return len(self.labels)
            
            
class ImageNet(data.Dataset):
    
    def __init__(self, data):
        
        self.dataset = data
