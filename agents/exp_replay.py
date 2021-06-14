import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
from importlib import import_module
from .default import NormalNN
from dataloaders.wrapper import Storage
from utils.metric import accuracy, AverageMeter, Timer
from collections import Counter


class NaiveRehearsal(NormalNN):

    def __init__(self, agent_config):
        super(NaiveRehearsal, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = agent_config['memory_size']
        self.task_memory = {}

        # override learn stream function to include replay
        def learn_stream(self, train_loader):
            # 1. Get the replay loader
            replay_list = []
            for storage in self.task_memory.values():
                replay_list.append(storage)
            replay_data = torch.utils.data.ConcatDataset(replay_list)

            # we want no. batches to be the same for the train and replay loaders
            n_train_batch = math.ceil(train_loader.dataset.len / train_loader.batch_size)
            replay_batch_size = math.ceil(replay_data.len / n_train_batch)
            # set up replay loader
            # we want the replay loader to be shuffled
            replay_loader = data.DataLoader(replay_data,
                                                        batch_size=replay_batch_size,
                                                        shuffle=True,
                                                        num_workers=train_loader.num_workers,
                                                        pin_memory=True)


            # 2. Update model, alternating between train batches and replay batches
            losses = AverageMeter()
            acc = AverageMeter()

            print('Batch\t Loss\t\t Acc')
            # iterating over train loader and replay loader
            for i, ((train_input, train_target),(replay_input, replay_target)) in enumerate(zip(train_loader, replay_loader)):
                # transferring to gpu if applicable
                if self.gpu:
                    train_input = train_input.cuda()
                    train_target = train_target.cuda()
                    replay_input = replay_input.cuda()
                    replay_target = replay_target.cuda()

                # getting loss, updating model on train batch
                train_loss, train_output = self.update_model(train_input, train_target)
                train_input = train_input.detach()
                train_target = train_target.detach()

                # getting loss, updating model on replay batch
                replay_loss, replay_output = self.update_model(replay_input, replay_target)
                replay_input = replay_input.detach()
                replay_target = replay_target.detach()

                # mask inactive output nodes
                train_output = train_output[:,self.active_out_nodes]
                replay_output = replay_output[:,self.active_out_nodes]

                # updating accuracy
                acc.update(accuracy(train_output, train_target), train_input.size(0))
                losses.update(train_loss, train_input.size(0))
                acc.update(accuracy(replay_output, replay_target), replay_input.size(0))
                losses.update(replay_loss, replay_input.size(0))

                print('[{0}/{1}]\t'
                              '{loss.val:.3f} ({loss.avg:.3f})\t'
                              '{acc.val:.2f} ({acc.avg:.2f})'.format(
                            i, len(train_loader), loss=losses, acc=acc))

            print(' * Train Acc: {acc.avg:.3f}'.format(acc=acc))

            # 3. Randomly decide which images to keep in memory
            self.task_count += 1
            # (a) Decide the number of samples to be saved
            num_sample_per_task = self.memory_size // self.task_count
            num_sample_per_task = min(len(train_loader.dataset), num_sample_per_task)
            # (b) Remove examples from memory to reserve space for new examples from latest task
            for storage in self.task_memory.values():
                storage.reduce(num_sample_per_task)
            # (c) Randomly choose some samples from the current task and save them to memory
            randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]
            self.task_memory[self.task_count] = Storage(train_loader.dataset, randind)



class GEM(NormalNN):
    """
    @inproceedings{GradientEpisodicMemory,
        title={Gradient Episodic Memory for Continual Learning},
        author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
        booktitle={NIPS},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}
    }
    """

    def __init__(self, agent_config):
        super(GEM, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = agent_config['memory_size']
        self.task_memory = {}
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.quadprog = import_module('quadprog')
        self.task_mem_cache = {}

    # storing all the gradients in a vector
    def grad_to_vector(self):
        vec = []
        for n, p in self.params.items():
            # storing gradients for parameters that have gradients
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            # filling zeroes for network parameters that have no gradient
            else:
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    # overwritting gradients with gradients stored in vector
    def vector_to_grad(self, vec):
        # overwrite the current param.grad by slicing the values in vec
        # flattening the gradient
        pointer = 0
        for n, p in self.params.items():
            # the length of the parameters
            num_param = p.numel()
            if p.grad is not None:
                # slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
            # increment the pointer
            pointer += num_param


    def project2cone2(self, gradient, memories):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector

            Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
        """
        # get te margin
        margin = self.config['reg_coef']
        # convert memories to numpy
        memories_np = memories.cpu().contiguous().double().numpy()
        gradient_np = gradient.cpu().contiguous().double().numpy()
        # t is the number of tasks
        t = memories_np.shape[0]

        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        P = P + G * 0.001
        h = np.zeros(t) + margin
        v = self.quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        new_grad = torch.Tensor(x).view(-1)
        if self.gpu:
            new_grad = new_grad.cuda()
        return new_grad


    def learn_stream(self, train_loader):
        print(self.task_memory)
        # update model as normal
        super(GEM, self).learn_stream(train_loader)

        self.task_memory[self.task_count] = train_loader.dataset
        self.task_count += 1

        # Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # concatentate all the data in each task
            mem_loader = data.DataLoader(mem,
                                         batch_size = len(mem),
                                         shuffle=False,
                                         num_workers=self.config['n_workers'],
                                         pin_memory=True)
            assert len(mem_loader)==1, 'The length of mem_loader should be 1'
            for i, (mem_input, mem_target) in enumerate(mem_loader):
                if self.gpu:
                    mem_input = mem_input.cuda()
                    mem_target = mem_target.cuda()
            self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':t}

    def update_model(self, out, targets):

        # compute gradients on previous tasks
        if self.task_count > 0:
            for t, mem in self.task_memory.items():
                self.zero_grad()
                # feed the data from memory and collect the gradients
                mem_out = self.forward(self.task_mem_cache[t]['data'])
                mem_loss = self.criterion(mem_out, self.task_mem_cache['target'])
                mem_loss.backward()
                # store the gradients
                self.task_grads[t] = self.grad_to_vector()

        # now compute the grad on current batch
        loss = self.criterion(out, targets)
        self.optimizer.zero_grad()
        loss.backward()

        # check if gradient violates constraints
        if self.task_count > 0:
            current_grad_vec = self.grad_to_vector()
            mem_grad_vec = torch.stack(list(self.task_grads.values()))
            dotp = current_grad_vec * mem_grad_vec
            dotp = dotp.sum(dim=1)
            if (dotp < 0).sum() != 0:
                new_grad = self.project2cone2(current_grad_vec, mem_grad_vec)
                # copy the gradients back
                self.vector_to_grad(new_grad)

        self.optimizer.step()
        return loss.detach()


class iCARL(NormalNN):
    """
    @inproceedings{iCARL,
        title={iCaRL: Incremental Classifier and Representation Learning},
        author={Rebuffi, Kolesnikov, Sperl, Lampert},
        booktitle={CVPR},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}
    }
    """

    def __init__(self, agent_config):

        # initializing model, optimizer, etc.
        super(iCARL, self).__init__(agent_config)

        self.memory_size = agent_config['memory_size']
        self.exemplars = {}
        self.seen_classes = 0
        self.dist_loss = torch.nn.BCELoss(reduction = 'mean')
        self.exemplar_means = None
        self.features = []
        # registering hook
        self.store_features = False
        #self.model.model.fc.register_forward_hook(self.hook_fn)
        #print(self.model.model[1])
        self.model.model[1].register_forward_hook(self.hook_fn)

    # hook function to get the input to a particular layer
    def hook_fn(self, module, input, output):
        if self.store_features:
            self.features.append(input[0].detach())

    def reset_features(self):
        self.features = []

    def criterion(self, pred, target, replay=False):
        # if it's a replay batch, use distillation loss
        if replay:
            # mask inactive output nodes
            pred = pred[:,self.active_out_nodes]
            target = target[:,self.active_out_nodes]
            # apply softmax, since outputs are currently logits
            pred = torch.sigmoid(pred)
            target = torch.sigmoid(target)
            # get distillation loss
            loss = self.dist_loss(pred, target)
            return loss

        # otherwise use regular nn loss
        else:
            loss = super(iCARL, self).criterion(pred, target)
            return loss

    def construct_exemplars(self, dataset, num_samples):

        # number of exemplars to store
        ex_size = min(num_samples, len(dataset))

        # compute features
        loader = data.DataLoader(dataset,
                                batch_size=100,
                                shuffle=False,
                                num_workers=self.config['n_workers'],
                                pin_memory=True)

        self.store_features = True
        self.reset_features()
        # iterate over loader, should be a single batch
        for i, (input, target) in enumerate(loader):
            # transfer to gpu if applicable
            if self.gpu:
                input = input.cuda()

            self.forward(input)

        # get features
        features = torch.cat(self.features, 0)
        # free self.features
        self.reset_features()
        self.store_features = False

        # normalize features
        for i in range(features.shape[0]):
            features[i] = features[i].data / features[i].norm()

        # getting mean & normalizing
        features = features.detach().cpu().numpy()
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)

        exemplar_set = []
        # list of tensors of shape (feature_size,)
        exemplar_features = []

        # computing exemplars
        for k in range(ex_size):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = (1.0 / (k+1)) * (phi + S)
            # normalize
            mu_p = mu_p / np.linalg.norm(mu_p)
            e = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

            exemplar_set.append(dataset[e][0])
            exemplar_features.append(features[e])

            #features = np.delete(features, e, axis = 0)

        exemplars = torch.stack(exemplar_set)

        exemplars = exemplars.detach().cpu()

        return exemplars

    # compute mean of exemplars in memory
    def compute_exemplar_means(self):
        exemplar_means = []
        self.reset_features()
        for y, P_y in self.exemplars.items():

            # wrapping P_y with dataloader, as the entire set of exemplars may not be able to fit on GPU
            loader = data.DataLoader(P_y,
                                     batch_size=100,
                                     shuffle=True,
                                     num_workers=self.config['n_workers'],
                                     pin_memory=True)
            # compute features
            for i, input in enumerate(loader):
                if self.gpu:
                    input = input.cuda()

                self.forward(input)

            # get features
            features = torch.cat(self.features, 0) # (batch_size, feature_size)
            self.reset_features()

            # taking the mean across the batch dimension
            mu_y = features.mean(0)
            # normalize
            mu_y = mu_y.data / mu_y.data.norm()
            exemplar_means.append(mu_y)

        self.exemplar_means = exemplar_means


    # performing nearest-mean-of-exemplars classification
    # transforming classification to a format that is being expected by validation function
    def predict(self, inputs):

        # return super(iCARL, self).predict(inputs)

        batch_size = inputs.size(0)

        # get exemplar means in desired format
        means = torch.stack(self.exemplar_means) # (n_classes, feature_size)
        means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
        means = means.transpose(1,2) # (batch_size, feature_size, n_classes)

        # calling forward places the features in the self.features variable, as we have registered a hook
        self.reset_features()
        self.forward(inputs)
        feature = torch.cat(self.features, 0)
        self.reset_features()

        # normalize predictions
        for i in range(feature.shape[0]):
            feature.data[i] = feature.data[i] / feature[i].norm()

        # reshape to proper size
        feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
        feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

        # get distance between y and mu_ys
        dists = (feature - means).pow(2).sum(1).squeeze() # (batch_size, n_classes)
        _, preds = dists.min(1)

        return (preds)


    # compute validation loss/accuracy
    # overriding because iCARl performs nearest-mean of exemplars classification
    def validation(self, dataloader):

        #return super(iCARL, self).validation(dataloader)

        acc = AverageMeter()
        batch_timer = Timer()
        batch_timer.tic()

        # keeping track of prior mode
        orig_mode = self.training

        self.eval()
        # computing mean of exemplars
        self.reset_features()
        self.store_features = True
        self.compute_exemplar_means()

        for i, (input, target) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()

            pred = self.predict(input)

            # computing accuracy
            accuracy = 100 * float((pred == target).sum()) / target.shape[0]

            acc.update(accuracy, input.size(0))

        # stop storing features
        self.store_features = False
        self.reset_features()

        # return model to original mode
        self.train(orig_mode)
        total_time = batch_timer.toc()

        return acc.avg, total_time


    def update_model(self, out, targets, replay=False):
        loss = self.criterion(out, targets, replay=replay)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()


    def learn_stream(self, train_loader):

        # if no classes have been seen yet, learn normally
        if self.seen_classes == 0:
            super(iCARL, self).learn_stream(train_loader)

        # else replay
        else:

            # 1. Get the replay loader
            replay_list = []
            for storage in self.exemplars.values():
                replay_list.append(storage)
            replay_data = torch.utils.data.ConcatDataset(replay_list)

            # we want no. batches to be the same for the train and replay loaders
            n_train_batch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
            replay_batch_size = math.ceil(len(replay_data) / n_train_batch)
            # set up replay loader
            # we want the replay loader to be shuffled
            replay_loader = data.DataLoader(replay_data,
                                            batch_size=replay_batch_size,
                                            shuffle=True,
                                            num_workers=train_loader.num_workers,
                                            pin_memory=True)

            print('Getting replay labels')

            # 2. Store pre-update model outputs on replay set
            replay_labels = {}
            for i, input in enumerate(replay_loader):
                # transferring to GPU if applicable
                if self.gpu:
                    input = input.cuda()

                out = self.forward(input)
                replay_labels[i] = out.detach().cpu()

            # 3. Update model, alternating between train batches and replay batches
            losses = AverageMeter()
            acc = AverageMeter()

            # initialize timers
            data_timer = Timer()
            batch_timer = Timer()
            train_timer = Timer()
            replay_timer = Timer()

            data_time = AverageMeter()
            batch_time = AverageMeter()
            train_time = AverageMeter()
            replay_time = AverageMeter()


            self.log('Batch\t Loss\t\t Acc')
            data_timer.tic()
            batch_timer.tic()
            # iterating over train loader and replay loader
            for i, ((train_input, train_target),(replay_input)) in enumerate(zip(train_loader, replay_loader)):
                # get replay target
                replay_target = replay_labels[i]
                # transferring to gpu if applicable
                if self.gpu:
                    train_input = train_input.cuda()
                    train_target = train_target.cuda()
                    replay_input = replay_input.cuda()
                    replay_target = replay_target.cuda()

                # measure data loading time
                data_time.update(data_timer.toc())

                # # for debugging purposes
                # print('Train input:')
                # print(train_input)
                # print('Train target:')
                # print(train_target)
                # print('Replay input:')
                # print(replay_input)
                # print('Replay target:')
                # print(replay_target)


                # forward pass
                train_timer.tic()
                train_output = self.forward(train_input)
                train_loss = self.update_model(train_output, train_target, replay = False)
                train_input = train_input.detach()
                train_target = train_target.detach()
                train_time.update(train_timer.toc())

                # regular loss, updating model on train batch
                # getting distillation loss, updating model on replay batch
                replay_timer.tic()
                replay_output = self.forward(replay_input)
                replay_loss = self.update_model(replay_output, replay_target, replay=True)
                replay_input = replay_input.detach()
                replay_target = replay_target.detach()
                replay_time.update(replay_timer.toc())

                # mask inactive output nodes
                train_output = train_output[:,self.active_out_nodes]

                # updating accuracy
                acc.update(accuracy(train_output, train_target), train_input.size(0))
                losses.update(train_loss, train_input.size(0))
                losses.update(replay_loss, replay_input.size(0))

                # measure elapsed time for entire batch
                batch_time.update(batch_timer.toc())
                # updating these timers with with current time
                data_timer.toc()
                train_timer.toc()
                replay_timer.toc()

                self.log('[{0}/{1}]\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), loss=losses, acc=acc))

            self.log(' * Train Acc: {acc.avg:.3f}'.format(acc=acc))
            self.log(' * Avg. Data time: {data_time.avg:.3f}, Avg. Batch time: {batch_time.avg:.3f}, Avg. Train-batch time: {train_time.avg:.3f}, Avg. Replay-batch time: {replay_time.avg:.3f}'
                 .format(data_time=data_time, batch_time=batch_time, train_time=train_time, replay_time=replay_time))


        # 3. Update Memory
        # number of unique classes in training dataset
        train_classes = Counter(train_loader.dataset.labels)
        seen_classes = set(range(self.seen_classes))
        new_classes = list(set(train_classes.keys()) - seen_classes)
        self.seen_classes += len(new_classes)
        # (a) Decide the number of samples to be saved
        num_sample_per_class = self.memory_size // self.seen_classes
        # (b) Remove examples from memory to reserve space for new examples from latest task
        for storage in self.exemplars.values():
            storage = storage[:num_sample_per_class]
        # (c) Construct exemplars for the newly seen classes
        for c in new_classes:
            # get subset of training dataset pertaining to this class
            inds = np.where(np.array(train_loader.dataset.labels) == c)[0].tolist()
            class_dataset = data.Subset(train_loader.dataset, inds)
            # construct exemplars
            print('Constructing exemplars for class ' + str(c))
            ex = self.construct_exemplars(class_dataset, num_sample_per_class)
            self.exemplars[c] = ex
        #print(self.exemplars)






















