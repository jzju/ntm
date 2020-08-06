import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from datetime import datetime

BATCH = 32
NSEQ = 2**10
SEQL = 16
DEV = "cuda"
logdir = "runs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = SummaryWriter(logdir)

seq = np.unique(np.random.randint(2, size=(NSEQ * 2, SEQL)), axis=0)
assert len(seq) >= NSEQ
np.random.shuffle(seq)
seq = seq[:NSEQ]
seq = np.hstack((seq, np.zeros((NSEQ, SEQL))))
ctl = np.zeros(SEQL * 2)
ctl[SEQL] = 1
seq = np.array([np.vstack((s, ctl)).T for s in seq])
seqt = seq[:NSEQ * 3 // 4]
seqv = seq[NSEQ * 3 // 4:NSEQ]



class Dataset(torch.utils.data.Dataset):
    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return seq[index]


class Net(nn.Module):
    def __init__(self, hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(2, hidden, 3, batch_first=True)
        self.res = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.h0 = torch.zeros(3, BATCH, self.hidden, device=DEV)
        self.c0 = torch.zeros(3, BATCH, self.hidden, device=DEV)

    def forward(self, x):
        y, _ = self.lstm(x, (self.h0, self.c0))
        y = y.contiguous().view(-1, self.hidden)  
        y = self.res(y)
        y = self.sigmoid(y) 
        y = y.view(BATCH, -1)
        return y


net = Net(256)
net.to(DEV)
opt = optim.Adam(net.parameters(), lr=0.00003)
crit = nn.BCELoss()


def rune(gen, train=True):
    ls = 0
    ll = 0
    la = 0
    with torch.set_grad_enabled(train):
        for x in gen:
            x = x.float()
            x = x.to(DEV)
            if train:
                opt.zero_grad()
            y = net(x)[:, SEQL:]
            yy = x[:, :SEQL, 0]
            loss = crit(y, yy)
            if train:
                loss.backward()
                opt.step()
            ls += SEQL * BATCH - torch.round(y).eq(yy).sum().item()
            # print(y)
            # print(torch.round(y))
            # la += y.eq(outputs.max(1)[1]).sum().item()
            ll += BATCH
    return ls / ll, yy.detach()[0], torch.round(y).detach()[0]


tds = torch.utils.data.DataLoader(Dataset(seqt), batch_size=BATCH, pin_memory=True, shuffle=True)
vds = torch.utils.data.DataLoader(Dataset(seqv), batch_size=BATCH, pin_memory=True, shuffle=True)
for e in range(10000):
    l, yy, y = rune(tds)
    writer.add_scalar('training loss', l, e)
    l, yy, y = rune(vds, False)
    writer.add_scalar('test loss', l, e)