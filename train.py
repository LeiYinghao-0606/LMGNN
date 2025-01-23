# train.py
from main import DataHandler
from LMGNN import MambaTransGNN_SelfSupervised
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from Params import args 


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # 设置随机种子以确保可重复性
    set_seed(42)

    # 初始化数据处理器并加载数据
    data_handler = DataHandler()
    data_handler.LoadData()

    # 初始化模型
    model = MambaTransGNN_SelfSupervised()

    # 如果有预训练模型，可以加载
    if args.load_model is not None:
        if os.path.exists(args.load_model):
            model.load_state_dict(torch.load(args.load_model, map_location=args.device))
            print(f"Loaded model from {args.load_model}")
        else:
            print(f"Model file {args.load_model} does not exist.")

    # 训练模型，传入 validation_set
    # 这里假设 validation_set 是通过 DataHandler 创建的 tstLoader
    model.train_model(data_handler, data_handler.tstLoader)

    # 保存模型
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'mamba_transgnn_model.pth'))
    print(f"Model saved to {os.path.join(args.save_path, 'mamba_transgnn_model.pth')}")

    # 进行预测或评估（根据需要）
    # user_embeds, item_embeds = model.predict(data_handler.torchBiAdj)
    # 进一步的操作...

if __name__ == "__main__":
    main()
