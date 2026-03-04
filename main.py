import torch as t
import torch.optim.lr_scheduler as lr_scheduler
import Utils.TimeLogger as logger
from Utils.TimeLogger import log as tlog
from Params import args
from model import LMGNN
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import setproctitle
import random
import time

# ===== NEW: epoch-only 日志（step 只终端单行刷新；仅 epoch 汇总写文件）=====
import json
import atexit
from datetime import datetime

_RUN_LOG_FH = None
_RUN_LOG_PATH = None


def _dataset_name(args_obj):
    """尽量从 args 中推断数据集名称。"""
    for k in ("dataset", "data", "data_name", "dataset_name", "dataname"):
        v = getattr(args_obj, k, None)
        if v is not None and str(v).strip():
            return str(v)
    return "unknown_dataset"


def _base_log_dir():
    """
    复用 TimeLogger 中原本定义的日志目录（若能取到）。
    取不到则回退到 ../Log
    """
    for attr in ("logPath", "log_dir", "logDir", "LOG_DIR", "path"):
        base = getattr(logger, attr, None)
        if isinstance(base, str) and base.strip():
            return os.path.normpath(base)
    return "../Log"


def _args_dict(args_obj):
    """将 args 转为可 JSON 序列化的 dict。"""
    d = dict(vars(args_obj)) if hasattr(args_obj, "__dict__") else {}

    def safe(v):
        try:
            json.dumps(v)
            return v
        except Exception:
            return str(v)

    return {k: safe(v) for k, v in d.items()}


def init_epoch_log(args_obj):
    """
    创建并写入日志头：
    <TimeLogger原始目录>/<dataset>/<save_path>_<timestamp>.log
    """
    global _RUN_LOG_FH, _RUN_LOG_PATH

    out_dir = os.path.join(_base_log_dir(), _dataset_name(args_obj))
    os.makedirs(out_dir, exist_ok=True)

    run_name = getattr(args_obj, "save_path", "run")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    _RUN_LOG_PATH = os.path.join(out_dir, f"{run_name}_{ts}.log")
    _RUN_LOG_FH = open(_RUN_LOG_PATH, "a", encoding="utf-8", buffering=1)

    # 让 TimeLogger 的输出目录也指向 dataset 子目录（即便后续不写文件，也更一致）
    if hasattr(logger, "logPath"):
        try:
            logger.logPath = out_dir + ("/" if not out_dir.endswith("/") else "")
        except Exception:
            pass

    _RUN_LOG_FH.write("=" * 90 + "\n")
    _RUN_LOG_FH.write(f"[RunStart] {datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
    _RUN_LOG_FH.write(f"[LogFile ] {_RUN_LOG_PATH}\n")
    _RUN_LOG_FH.write(f"[Dataset ] {_dataset_name(args_obj)}\n")
    _RUN_LOG_FH.write("[Args]\n")
    _RUN_LOG_FH.write(json.dumps(_args_dict(args_obj), ensure_ascii=False, indent=2) + "\n")
    _RUN_LOG_FH.write("=" * 90 + "\n")

    # 终端提示一次（不写 TimeLogger 默认日志）
    tlog(f"Epoch-only log enabled: {_RUN_LOG_PATH}", save=False, oneline=False)

    atexit.register(close_epoch_log)


def close_epoch_log():
    global _RUN_LOG_FH
    if _RUN_LOG_FH is not None:
        try:
            _RUN_LOG_FH.flush()
            _RUN_LOG_FH.close()
        except Exception:
            pass
        _RUN_LOG_FH = None


def _write_epoch_line(msg: str):
    if _RUN_LOG_FH is None:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _RUN_LOG_FH.write(f"[{ts}] {msg}\n")


def log(msg, save=True, oneline=False):
    """
    统一日志入口：
    - 终端输出：仍使用 TimeLogger 的打印能力（支持 oneline 单行刷新）
    - 文件写入：仅当 save=True 且 oneline=False 时写入（即 epoch 汇总/关键事件）
    """
    # 关键：强制 TimeLogger 不落盘，避免重复/混乱
    tlog(msg, save=False, oneline=oneline)

    if save and (not oneline):
        _write_epoch_line(str(msg))


# === 参数统计与显存估算工具 ===
def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def param_memory_bytes(model):
    # 仅参数本体（不含梯度/优化器状态）
    return sum(p.numel() * p.element_size() for p in model.parameters())


def pretty_size(num_bytes: int):
    num = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} EB"


def top_k_tensors(model, k=10):
    items = []
    for name, p in model.named_parameters():
        items.append((name, p.numel(), tuple(p.shape), p.dtype))
    items.sort(key=lambda x: x[1], reverse=True)
    lines = []
    for name, n, shape, dt in items[:k]:
        lines.append(f"{name:50s} | {n:12,} | {str(shape):>18s} | {str(dt)}")
    return "\n".join(lines)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)  # 如果使用多GPU
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def sample_nodes_by_degree(degrees, top_perc=0.05, bottom_perc=0.80, sample_num=500):
    n = len(degrees)
    sorted_idx = np.argsort(degrees)
    top_k = max(1, int(n * top_perc))
    bottom_k = max(1, int(n * bottom_perc))
    hot_idx = np.random.choice(sorted_idx[-top_k:], size=min(sample_num, top_k), replace=False)
    cold_idx = np.random.choice(sorted_idx[:bottom_k], size=min(sample_num, bottom_k), replace=False)
    return hot_idx, cold_idx


class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.user_degrees = self.handler.user_pop.detach().cpu().numpy()
        self.item_degrees = self.handler.item_pop.detach().cpu().numpy()

        log(f"USER {args.user} ITEM {args.item}")
        log(f"NUM OF INTERACTIONS {self.handler.trnLoader.dataset.__len__()}")

        self.metrics = dict()
        mets = ["Loss", "preLoss", "Recall", "NDCG"]
        for met in mets:
            self.metrics["Train" + met] = list()
            self.metrics["Test" + met] = list()

        self.metrics["TrainTime"] = list()
        self.metrics["TestTime"] = list()

        # 预留模型信息容器（将随历史一并保存）
        self.metrics["ModelInfo"] = {}

        # Early stopping state
        self.best_metric = float("-inf")
        self.no_improve = 0

    def makePrint(self, name, ep, reses, save, epoch_time=None):
        ret = "Epoch %d/%d, %s: " % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += "%s = %.4f, " % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        if epoch_time is not None:
            ret += "%sTime = %.2f sec, " % (name, epoch_time)
            tem_time = name + "Time"
            if save and tem_time in self.metrics:
                self.metrics[tem_time].append(epoch_time)
        ret = ret[:-2] + "  "
        return ret

    def run(self):
        self.prepareModel()
        log("Model Prepared")

        self.user_pop = self.user_pop.to(args.device)
        self.item_pop = self.item_pop.to(args.device)
        self.model.set_node_degree(self.user_pop, self.item_pop)

        if args.load_model is not None:
            self.loadModel()
            stloc = len(self.metrics["TrainLoss"]) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log("Model Initialized")

        for ep in range(stloc, args.epoch):
            if hasattr(self.model, "set_noise_anneal"):
                self.model.set_noise_anneal(ep, args.epoch)

            tstFlag = (ep % args.tstEpoch == 0)

            # 训练
            reses, train_elapsed = self.trainEpoch()
            log(self.makePrint("Train", ep, reses, tstFlag, epoch_time=train_elapsed))

            # 测试
            if tstFlag:
                reses, test_elapsed = self.testEpoch()
                log(self.makePrint("Test", ep, reses, tstFlag, epoch_time=test_elapsed))
                self.saveHistory()

                # Early stopping on primary metric
                metric_key = f"Recall@{args.topk[0]}"
                cur_metric = reses.get(metric_key, None)
                if cur_metric is not None:
                    if cur_metric > self.best_metric + args.early_stop_min_delta:
                        self.best_metric = cur_metric
                        self.no_improve = 0
                    else:
                        self.no_improve += 1
                        if args.early_stop_patience > 0 and self.no_improve >= args.early_stop_patience:
                            log(f"Early stop at epoch {ep}: {metric_key} no improve for {self.no_improve} evals")
                            break

            current_lr = self.opt.param_groups[0]["lr"]
            log(f"Current Learning Rate: {current_lr}")

        # 训练结束后再做一次测试
        reses, test_elapsed = self.testEpoch()
        log(self.makePrint("Test", args.epoch, reses, True, epoch_time=test_elapsed))
        self.saveHistory()

    def prepareModel(self):
        self.model = LMGNN().cuda()
        # 避免与损失里的 L2 正则重复，这里不设置 weight_decay
        self.opt = t.optim.AdamW(self.model.parameters(), lr=args.lr)

        # 统计参数与显存，并记录到 epoch 日志
        total_params = count_params(self.model, trainable_only=False)
        trainable_params = count_params(self.model, trainable_only=True)
        param_bytes = param_memory_bytes(self.model)
        grads_bytes = param_bytes
        adam_states_bytes = 2 * param_bytes
        approx_train_bytes = param_bytes + grads_bytes + adam_states_bytes

        log(f"Model Params: total={total_params:,}, trainable={trainable_params:,}")
        log(f"Parameter memory: {pretty_size(param_bytes)}")
        log(f"~Training-time memory (params + grads + Adam): {pretty_size(approx_train_bytes)}")

        topk = getattr(args, "param_topk", 0)
        if topk and topk > 0:
            log(f"Top-{topk} largest parameter tensors:\n{top_k_tensors(self.model, k=topk)}")

        self.metrics["ModelInfo"] = {
            "TotalParams": int(total_params),
            "TrainableParams": int(trainable_params),
            "ParamMemoryBytes": int(param_bytes),
            "ApproxTrainMemoryBytes": int(approx_train_bytes),
            "HumanReadable": {
                "ParamMemory": pretty_size(param_bytes),
                "ApproxTrainMemory": pretty_size(approx_train_bytes),
            },
        }

        self.user_pop = self.handler.user_pop.to(args.device)
        self.item_pop = self.handler.item_pop.to(args.device)
        if args.cl_pop_beta != 1.0:
            self.user_pop = self.user_pop.clamp_min(1e-6).pow(args.cl_pop_beta)
            self.item_pop = self.item_pop.clamp_min(1e-6).pow(args.cl_pop_beta)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        total_train_time = 0.0

        start_event = t.cuda.Event(enable_timing=True)
        end_event = t.cuda.Event(enable_timing=True)

        self.model.train()
        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            start_event.record()

            user_weight = self.user_pop.index_select(0, ancs)
            item_weight = self.item_pop.index_select(0, poss)
            bprLoss = self.model.calcLosses(
                ancs,
                poss,
                negs,
                self.handler.torchBiAdj,
                user_weight=user_weight,
                item_weight=item_weight,
            )
            loss = bprLoss

            self.opt.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.opt.step()

            end_event.record()
            t.cuda.synchronize()

            elapsed = start_event.elapsed_time(end_event) / 1000.0
            total_train_time += elapsed

            epLoss += loss.item()
            epPreLoss += bprLoss.item()

            # step 信息：终端单行刷新，不写入文件
            log("Step %d/%d: loss = %.3f      " % (i, steps, float(loss.item())), save=False, oneline=True)

        ret = dict()
        denom = max(steps, 1)
        ret["Loss"] = epLoss / denom
        ret["preLoss"] = epPreLoss / denom

        return ret, total_train_time

    def testEpoch(self):
        """
        根据多种 k (args.topk) 计算 Recall@k、NDCG@k
        """
        tstLoader = self.handler.tstLoader
        k_list = args.topk
        max_k = max(k_list)

        epRecall_dict = {k: 0.0 for k in k_list}
        epNdcg_dict = {k: 0.0 for k in k_list}
        total_test_time = 0.0

        i = 0
        num = len(tstLoader.dataset)
        steps = num // args.tstBat

        start_event = t.cuda.Event(enable_timing=True)
        end_event = t.cuda.Event(enable_timing=True)

        self.model.eval()
        with t.no_grad():
            usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)

        for batch in tstLoader:
            i += 1
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                usr, trnMask = batch
                usr = usr.long().cuda()
                trnMask = trnMask.cuda()
                trn_pos_list = None
                tst_pos_list = None
            else:
                usr, trn_pos_list, tst_pos_list = batch
                usr = usr.long().cuda()

            start_event.record()

            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0))
            if trn_pos_list is not None:
                for row_idx, trn_pos in enumerate(trn_pos_list):
                    if trn_pos.numel() > 0:
                        allPreds[row_idx, trn_pos] = -1e8
            else:
                allPreds = allPreds * (1 - trnMask) - trnMask * 1e8

            _, topLocs = t.topk(allPreds, max_k, dim=1)

            end_event.record()
            t.cuda.synchronize()

            elapsed = start_event.elapsed_time(end_event) / 1000.0
            total_test_time += elapsed

            for k in k_list:
                topLocs_k = topLocs[:, :k]
                if tst_pos_list is not None:
                    recall_k, ndcg_k = self.calcResFromList(topLocs_k.cpu().numpy(), tst_pos_list, k)
                else:
                    recall_k, ndcg_k = self.calcRes(
                        topLocs_k.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, k
                    )
                epRecall_dict[k] += recall_k
                epNdcg_dict[k] += ndcg_k

            # 测试进度：终端单行刷新，不写入文件
            info_str = f"Steps {i}/{steps}: "
            for k in k_list:
                avg_recall = epRecall_dict[k] / (i * args.tstBat)
                avg_ndcg = epNdcg_dict[k] / (i * args.tstBat)
                info_str += f"R@{k}={avg_recall:.4f}, N@{k}={avg_ndcg:.4f}; "
            log(info_str, save=False, oneline=True)

        ret = dict()
        for k in k_list:
            ret[f"Recall@{k}"] = epRecall_dict[k] / num
            ret[f"NDCG@{k}"] = epNdcg_dict[k] / num

        return ret, total_test_time

    def calcRes(self, topLocs, tstLocs, batIds, K):
        assert topLocs.shape[0] == len(batIds)
        allRecall = 0.0
        allNdcg = 0.0

        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            if tstNum == 0:
                continue

            maxDcg = np.sum([1.0 / np.log2(j + 2) for j in range(min(tstNum, K))])
            recall = 0
            dcg = 0.0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    idx = temTopLocs.index(val)
                    dcg += 1.0 / np.log2(idx + 2)

            recall = recall / tstNum
            ndcg = (dcg / maxDcg) if maxDcg > 0 else 0.0
            allRecall += recall
            allNdcg += ndcg

        return allRecall, allNdcg

    def calcResFromList(self, topLocs, tst_pos_list, K):
        allRecall = 0.0
        allNdcg = 0.0

        for i in range(len(tst_pos_list)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tst_pos_list[i].cpu().numpy().tolist()
            tstNum = len(temTstLocs)
            if tstNum == 0:
                continue

            maxDcg = np.sum([1.0 / np.log2(j + 2) for j in range(min(tstNum, K))])
            recall = 0
            dcg = 0.0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    idx = temTopLocs.index(val)
                    dcg += 1.0 / np.log2(idx + 2)

            recall = recall / tstNum
            ndcg = (dcg / maxDcg) if maxDcg > 0 else 0.0
            allRecall += recall
            allNdcg += ndcg

        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return

        history_dir = "../History"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        history_path = os.path.join(history_dir, args.save_path + ".his")
        with open(history_path, "wb") as fs:
            pickle.dump(self.metrics, fs)

        models_dir = "../Models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, args.save_path + ".mod")
        content = {
            "model": self.model,
            "optimizer": self.opt.state_dict(),
        }
        t.save(content, model_path)
        log("Model Saved: %s" % args.save_path)

    def loadModel(self):
        ckp = t.load("../Models/" + args.load_model + ".mod")
        self.model = ckp["model"]
        self.opt = t.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.opt.load_state_dict(ckp["optimizer"])

        with open("../History/" + args.load_model + ".his", "rb") as fs:
            self.metrics = pickle.load(fs)
        log("Model Loaded")


if __name__ == "__main__":
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setproctitle.setproctitle("proc_title")

    # 关键：关闭 TimeLogger 自带的落盘机制，避免重复/混乱
    logger.saveDefault = False
    logger.logPath = "/root/LMGNN/History"  
    # 启用 epoch-only 文件日志：<TimeLogger原始目录>/<dataset>/<save_path>_<time>.log
    init_epoch_log(args)

    log("Start")
    handler = DataHandler()
    handler.LoadData()
    log("Load Data")
    coach = Coach(handler)
    coach.run()
