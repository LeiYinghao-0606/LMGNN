import pickle
import numpy as np
from scipy.sparse import coo_matrix, spmatrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
from torch.utils.data import WeightedRandomSampler

class DataHandler:
    def __init__(self):
        if args.data == 'yelp':
            predir = 'Data/yelp/'
        elif args.data == 'ml-10m':
            predir = 'Data/ml-10m/'
        elif args.data == 'tmall':
            predir = 'Data/tmall/'
        elif args.data == 'amazon-books':
            predir = 'Data/amazon-books/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse_coo_tensor(idxs, vals, shape, dtype=t.float32, device='cuda')

    def makeSample(self):
        user_sample_idx = t.tensor([[args.user + i for i in range(args.item)] * args.user])
        item_sample_idx = t.tensor([[i for i in range(args.user)] * args.item])
        return user_sample_idx, item_sample_idx

    def makeMask(self):
        u_u_mask = t.zeros(size=(args.user, args.user), dtype=bool)
        u_i_mask = t.ones(size=(args.user, args.item), dtype=bool)

        i_i_mask = t.zeros(size=(args.item, args.item), dtype=bool)
        i_u_mask = t.ones(size=(args.item, args.user), dtype=bool)

        u_mask = t.concat([u_u_mask, u_i_mask], dim=-1)
        i_mask = t.concat([i_u_mask, i_i_mask], dim=-1)

        mask = t.concat([u_mask, i_mask], dim=0)
        return mask

    def LoadData(self):
        trnMat = self.loadOneFile(self.trnfile)
        tstMat = self.loadOneFile(self.tstfile)
        args.user, args.item = trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.mask = self.makeMask()
        user_degrees = np.array(trnMat.sum(axis=1)).flatten()  # shape: (user,)
        low_th = np.percentile(user_degrees, 33)
        high_th = np.percentile(user_degrees, 66)

        def assign_stratum(deg):
            if deg <= low_th:
                return 0
            elif deg <= high_th:
                return 1
            else:
                return 2

        user_strata = np.array([assign_stratum(d) for d in user_degrees])

        trnData = TrnData(trnMat)

        stratum_weights = {0: 2.0, 1: 1.5, 2: 1.0} 
        sample_weights = []
        for idx in range(len(trnData)):
            u, i, neg = trnData[idx]
            s = user_strata[u]
            w = stratum_weights[s]
            sample_weights.append(w)

        sample_weights = t.DoubleTensor(sample_weights)

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        self.trnLoader = dataloader.DataLoader(
            trnData, 
            batch_size=args.batch, 
            sampler=sampler,  
            shuffle=False,    
            num_workers=4, 
            pin_memory=True
        )

        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=4, pin_memory=True)


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.negSampling()

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
