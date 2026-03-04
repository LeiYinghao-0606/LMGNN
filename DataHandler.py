import pickle
import numpy as np
import scipy.sparse as sp
from Params import args
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import os

# ====== 可选：是否强行兼容旧评测的 dense 返回（会很占内存，谨慎打开） ====== #
LEGACY_DENSE_TEST_ROW = True  # True=返回 (u, dense_row)，False=返回索引并用 collate 处理

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
        # 统一成 COO & 二值 float32
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
        # make ui adj (保持稀疏，不 densify)
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        bi = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        bi = (bi != 0).astype(np.float32)
        bi = self.normalizeAdj(bi).tocoo()

        idxs = t.from_numpy(np.vstack([bi.row, bi.col]).astype(np.int64))
        vals = t.from_numpy(bi.data.astype(np.float32))
        shape = t.Size(bi.shape)
        # 若你原模型要求在 GPU 上，这里也可以 .cuda()，但建议延迟到用到时 .to(device)
        return t.sparse_coo_tensor(idxs, vals, shape).cuda()

    # —— 兼容：保留接口，但避免巨大 dense mask —— #
    def makeSample(self):
        # 旧函数是 O(U*I) 的索引表，内存炸弹；现在返回 None, None 以保持接口存在
        return None, None

    def makeMask(self):
        # 旧函数会构造 (U+I)^2 的 dense 布尔矩阵，直接废弃，返回 None
        return None

    def LoadData(self):
        trnMat = self.loadOneFile(self.trnfile)  # [U, I] COO
        tstMat = self.loadOneFile(self.tstfile)  # [U, I] COO
        args.user, args.item = trnMat.shape

        self.user_pop = t.from_numpy(np.array(trnMat.sum(axis=1)).flatten().astype(np.float32))
        self.item_pop = t.from_numpy(np.array(trnMat.sum(axis=0)).flatten().astype(np.float32))

        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.mask = self.makeMask()  # 现在是 None，仅保持属性存在，避免其他地方 AttributeError

        # 训练数据（支持 .negSampling() 的离线/在线二合一实现）
        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(
            trnData,
            batch_size=args.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        # 测试数据（默认返回索引；如需旧评测兼容，打开上面的 LEGACY_DENSE_TEST_ROW）
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
                collate_fn=tst_collate  # 处理变长索引
            )

# =========================
# 训练数据集：支持 .negSampling()（可选）+ 在线兜底
# =========================
class TrnData(data.Dataset):
    def __init__(self, coomat: sp.coo_matrix):
        self.rows = coomat.row.astype(np.int64)
        self.cols = coomat.col.astype(np.int64)
        self.dokmat = coomat.todok()
        self.negs = None  # 懒加载；调用 negSampling() 后才会填充

    # 兼容你的训练循环：每个 epoch 会调用 .negSampling()
    def negSampling(self):
        # 若你担心时间或内存，这里也可以留空（pass）改为完全在线
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
            # 在线负采样兜底
            while True:
                j = np.random.randint(args.item)
                if (u, j) not in self.dokmat:
                    break
        return u, i, j

# =========================
# 测试数据集：默认返回索引（历史 + 测试正例）
# 如需强行旧兼容，legacy_dense=True 返回 (u, dense_row) —— 占内存！
# =========================
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
            # —— 旧兼容（占内存）：返回训练历史的 dense 行 —— #
            dense_row = np.zeros(self.csr_trn.shape[1], dtype=np.float32)
            dense_row[self.csr_trn[u].indices] = 1.0
            return u, dense_row
        else:
            # —— 推荐：返回索引 —— #
            trn_pos_items = self.csr_trn[u].indices.astype(np.int64)   # 历史交互
            tst_pos_items = np.array(self.tstLocs[u], dtype=np.int64)  # 测试正例
            return u, trn_pos_items, tst_pos_items

# =========================
# 测试集 collate（索引模式）
# =========================
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
