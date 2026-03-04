import pickle
import numpy as np
import scipy.sparse as sp
from Params import args
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import os

LEGACY_DENSE_TEST_ROW = True  

class DataHandler:
    def __init__(self):
        if args.data == 'yelp':
            predir = 'Data/yelp/'
        elif args.data == 'yelp2018':
            predir = 'Data/yelp2018/'
        elif args.data == 'ml-10m':
            predir = 'Data/ml-10m/'
        elif args.data == 'tmall':
            predir = 'Data/tmall/'
        elif args.data == 'gowalla':
            predir = 'Data/gowalla/'
        elif args.data == 'amazon-books':
            predir = 'Data/amazon-books/'
        elif args.data == 'gowalla':
            predir = 'Data/gowalla/'
        else:
            raise ValueError(f"Unknown dataset: {args.data}")
        self.predir = predir
        self.trnfile = os.path.join(predir, 'trnMat.pkl')
        self.tstfile = os.path.join(predir, 'tstMat.pkl')

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            mat = pickle.load(fs)
        if not sp.isspmatrix_coo(mat):
            mat = mat.tocoo()
        data = (mat.data != 0).astype(np.float32)
        return sp.coo_matrix((data, (mat.row, mat.col)), shape=mat.shape)

    def normalizeAdj(self, mat: sp.coo_matrix):
        degree = np.array(mat.sum(axis=-1)).flatten()
        dInvSqrt = np.power(degree + 1e-7, -0.5)
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrt[np.isnan(dInvSqrt)] = 0.0
        d_mat_inv = sp.diags(dInvSqrt)
        normalized_adj = d_mat_inv.dot(mat).dot(d_mat_inv).tocoo()
        return normalized_adj

    def makeTorchAdj(self, mat: sp.coo_matrix):
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        bi = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        bi = (bi != 0).astype(np.float32)
        bi = self.normalizeAdj(bi).tocoo()

        idxs = t.from_numpy(np.vstack([bi.row, bi.col]).astype(np.int64))
        vals = t.from_numpy(bi.data.astype(np.float32))
        shape = t.Size(bi.shape)
        return t.sparse_coo_tensor(idxs, vals, shape).cuda()

    def makeSample(self):
        return None, None

    def makeMask(self):
        return None

    def LoadData(self):
        trnMat = self.loadOneFile(self.trnfile)  # [U, I] COO
        tstMat = self.loadOneFile(self.tstfile)  # [U, I] COO
        args.user, args.item = trnMat.shape

        self.user_pop = t.from_numpy(np.array(trnMat.sum(axis=1)).flatten().astype(np.float32))
        self.item_pop = t.from_numpy(np.array(trnMat.sum(axis=0)).flatten().astype(np.float32))

        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.mask = self.makeMask()  

        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(
            trnData,
            batch_size=args.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        tstData = TstData(tstMat, trnMat, legacy_dense=LEGACY_DENSE_TEST_ROW)
        if LEGACY_DENSE_TEST_ROW:
            self.tstLoader = dataloader.DataLoader(
                tstData,
                batch_size=args.tstBat,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        else:
            self.tstLoader = dataloader.DataLoader(
                tstData,
                batch_size=args.tstBat,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=tst_collate  
            )


class TrnData(data.Dataset):
    def __init__(self, coomat: sp.coo_matrix):
        self.rows = coomat.row.astype(np.int64)
        self.cols = coomat.col.astype(np.int64)
        self.dokmat = coomat.todok()
        self.negs = None  


    def negSampling(self):
        self.negs = np.empty(len(self.rows), dtype=np.int64)
        for i in range(len(self.rows)):
            u = int(self.rows[i])
            while True:
                j = np.random.randint(args.item)
                if (u, j) not in self.dokmat:
                    self.negs[i] = j
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        u = int(self.rows[idx])
        i = int(self.cols[idx])
        if self.negs is not None:
            j = int(self.negs[idx])
        else:
            while True:
                j = np.random.randint(args.item)
                if (u, j) not in self.dokmat:
                    break
        return u, i, j

class TstData(data.Dataset):
    def __init__(self, coomat: sp.coo_matrix, trnMat: sp.coo_matrix, legacy_dense: bool = False):
        self.legacy_dense = legacy_dense
        self.csr_trn = (trnMat.tocsr() != 0).astype(np.int8)

        U = coomat.shape[0]
        tstLocs = [None] * U
        tstUsrs = set()
        rows, cols = coomat.row.astype(np.int64), coomat.col.astype(np.int64)
        for r, c in zip(rows, cols):
            if tstLocs[r] is None:
                tstLocs[r] = []
            tstLocs[r].append(int(c))
            tstUsrs.add(int(r))

        self.tstUsrs = np.array(sorted(list(tstUsrs)), dtype=np.int64)
        self.tstLocs = [lst or [] for lst in tstLocs]

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        u = int(self.tstUsrs[idx])
        if self.legacy_dense:
            dense_row = np.zeros(self.csr_trn.shape[1], dtype=np.float32)
            dense_row[self.csr_trn[u].indices] = 1.0
            return u, dense_row
        else:
            trn_pos_items = self.csr_trn[u].indices.astype(np.int64)   
            tst_pos_items = np.array(self.tstLocs[u], dtype=np.int64)  
            return u, trn_pos_items, tst_pos_items

def tst_collate(batch):
    # batch: list of (u: int, trn_pos_items: np.ndarray, tst_pos_items: np.ndarray)
    users = []
    trn_pos_list = []
    tst_pos_list = []
    for u, trn_pos, tst_pos in batch:
        users.append(u)
        trn_pos_list.append(t.as_tensor(trn_pos, dtype=t.int64))
        tst_pos_list.append(t.as_tensor(tst_pos, dtype=t.int64))
    return t.as_tensor(users, dtype=t.int64), trn_pos_list, tst_pos_list
