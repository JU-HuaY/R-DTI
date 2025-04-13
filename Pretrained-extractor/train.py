import os
import torch
import torch.nn as nn
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from data_loader import Protein_pkl_Dataset
from data_loader import my_collate
from Representation import Representation_model
import torch.optim as optim
import timeit


class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)
        self.batch_size = batch_size
        # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        loss_total, loss1_, loss2_, loss3_ = 0, 0, 0, 0
        self.optimizer.zero_grad()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                collate_fn=my_collate)
        for data in dataloader:
            loss_all, loss1, loss2, loss3 = model(data, device)
            loss_all.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_total += loss_all.mean().item()
            loss1_ += loss1.mean().item()
            loss2_ += loss2.mean().item()
            loss3_ += loss3.mean().item()
            # loss_total2 += loss3.item()
        return loss_total, loss1_, loss2_, loss3_


class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset):
        loss_total, loss1_, loss2_, loss3_ = 0, 0, 0, 0

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                collate_fn=my_collate)
        for data in dataloader:
            loss_all, loss1, loss2, loss3 = model(data, device)
            loss_total += loss_all.mean().item()
            loss1_ += loss1.mean().item()
            loss2_ += loss2.mean().item()
            loss3_ += loss3.mean().item()

        return loss_total, loss1_, loss2_, loss3_


    def save_Losses(self, Losses, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, Losses)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    batchs = 12
    lr = 1e-3
    weight_decay = 0.1
    iteration = 120
    decay_interval = 5
    lr_decay = 0.5
    setting = "ssl_LM"
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    train_dataset = Protein_pkl_Dataset(root_dir='train')
    validation_dataset = Protein_pkl_Dataset(root_dir='validation')
    model = Representation_model(3, 128, 256, 64, batchs)
    model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
    trainer = Trainer(model, batchs, lr, weight_decay)
    tester = Tester(model, batchs)
    """Output files."""
    file_Losses = 'output/result/loss--' + setting + '.txt'
    file_model = 'output/model/' + setting
    Losses = ('Epoch\tTime(sec)\tLoss_all\tLoss_1\tLoss_2\tLoss_3\t'
              'loss_test\tloss1_test\tloss2_test\tloss3_test')
    with open(file_Losses, 'w') as f:
        f.write(Losses + '\n')

    """Start training."""
    print('Training...')
    print(Losses)
    start = timeit.default_timer()
    loss_totals = 1000
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_total, loss1_, loss2_, loss3_ = trainer.train(train_dataset)
        loss_test, loss1_test, loss2_test, loss3_test = tester.test(validation_dataset)
        end = timeit.default_timer()
        time = end - start
        Losses = [epoch, time, loss_total, loss1_, loss2_, loss3_, loss_test, loss1_test, loss2_test, loss3_test]
        print('\t'.join(map(str, Losses)))
        tester.save_Losses(Losses, file_Losses)
        if loss_test < loss_totals:
            loss_totals = loss_test
            tester.save_model(model, file_model)


