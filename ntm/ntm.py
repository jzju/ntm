import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from datetime import datetime


def ps(x):
    print(x.size())


BATCH = 1
NSEQ = 2**10
SEQL = 16
DEV = "cpu"
logdir = "runs/" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "/"
writer = SummaryWriter(logdir)

N = 128
M = 20
CZ = 32

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


class LSTM(nn.Module):
    def __init__(self, hidden):
        super(LSTM, self).__init__()
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


# def roll(x, n):  
#     return torch.cat((x[:, -n:], x[:, :-n]), dim=1)


def roll(x, n):  
    return torch.cat((x[-n:], x[:-n]))


class HEAD(nn.Module):
    def __init__(self, read):
        super(HEAD, self).__init__()
        self.ww = nn.Parameter(torch.rand(N, device=DEV))
        self.ctlk = nn.Sequential(nn.Linear(CZ, M))
        self.ctlb = nn.Sequential(nn.Linear(CZ, 1), nn.Softplus())
        self.ctlg = nn.Sequential(nn.Linear(CZ, 1), nn.Sigmoid())
        self.ctls = nn.Sequential(nn.Linear(CZ, 3), nn.Softmax(dim=0))
        self.ctly = nn.Sequential(nn.Linear(CZ, 1), nn.Softplus())
        self.read = read
        if not read:
            self.e = nn.Sequential(nn.Linear(CZ, M), nn.Sigmoid())
            self.a = nn.Sequential(nn.Linear(CZ, M))
    
    def reset(self):
        self.w = self.ww.clone()
    
    def forward(self, cin, mem):
        k = self.ctlk(cin)
        b = self.ctlb(cin)
        g = self.ctlg(cin)
        s = self.ctls(cin)
        y = self.ctly(cin) + 1
        cos = nn.CosineSimilarity(dim=1)
        wc = cos(mem, k.unsqueeze(0).expand(N, M))
        wc = F.softmax(wc * b, dim=0)
        wg = wc * g + (1 - g) * self.w
        wm = roll(wg, -1)
        wp = roll(wg, 1)
        wt = s[0] * wm + s[1] * wg + s[2] * wp
        wt = F.normalize(wt ** y, dim=0, p=1)
        self.w = wt
        if self.read:
            return wt.matmul(mem)
        else:
            e = wt.unsqueeze(1).matmul(self.e(cin).unsqueeze(0))
            mem = mem * (1 - e)
            a = wt.unsqueeze(1).matmul(self.a(cin).unsqueeze(0))
            mem = mem + a


class NTMFF(nn.Module):
    def __init__(self):
        super(NTMFF, self).__init__()
        self.MM = nn.Parameter(torch.rand(N, M, device=DEV))
        self.rh = HEAD(1)
        self.wh = HEAD(0)
        self.cin = nn.Sequential(nn.Linear(2, 64), nn.Linear(64, CZ))
        self.cout = nn.Sequential(nn.Linear(M, 1), nn.Sigmoid())
    
    def reset(self):
        self.M = self.MM.clone()
        self.rh.reset()
        self.wh.reset()

    def forward(self, xs):
        res = []
        for x in xs:
            cin = self.cin(x)
            r = self.rh(cin, self.M)
            r = self.cout(r)
            self.wh(cin, self.M)
            res.append(r)
            
        return torch.cat(res)

net = NTMFF()
# net = LSTM(256)
net.to(DEV)
opt = optim.Adam(net.parameters(), lr=0.0001)
crit = nn.BCELoss()


def rune(gen, train=True):
    ls = 0
    ll = 0
    la = 0
    i = 0
    with torch.set_grad_enabled(train):
        for x in gen:
            x = x.float()
            x = x.to(DEV)
            x = x.view(-1, 2)
            if train:
                opt.zero_grad()
            net.reset()
            y = net(x)[SEQL:]
            # if i % 100 == 0:
            #     print(y)
            i += 1
            yy = x[:SEQL, 0]
            # print(y)
            # print(yy)
            try:
                loss = crit(y, yy)
            except:
                print(y)
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
    print(l, e)
    writer.add_scalar('training loss', l, e)
    # l, yy, y = rune(vds, False)
    # writer.add_scalar('test loss', l, e)