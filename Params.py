import argparse
def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')  # 调低学习率
    parser.add_argument('--batch', default=16384, type=int, help='batch size')  # 减小批次大小
    parser.add_argument('--device',default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--tstBat', default=128, type=int, help='number of users in a testing batch')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')  # 增加训练轮数
    parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
    parser.add_argument('--latdim', default=128, type=int, help='embedding size')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--topk', default=20, type=int, help='K of top K')
    parser.add_argument('--num_gnn_layers', default=4, type=float, help='number of gnn layers')
    parser.add_argument('--num_mamba_layers', default=4, type=float, help='number of mamba layers')
    parser.add_argument('--data', default='tmall', type=str, help='name of dataset')
    parser.add_argument('--tstEpoch', default=2, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')  
    parser.add_argument('--dropout', default=0.4, type=float, help='Ratio of transformer layer dropout')  # 增加 dropout
    parser.add_argument('--layer_cl', default=3, type=int, help='进行对比学习的层数，对应代码中的 `layer_cl')  # 增加对比学习层数
    parser.add_argument('--lambda_cl', default=0.2, type=float, help='对比学习的权重，与 XSimGCL 中的 `lambda')  # 增加对比损失权重
    parser.add_argument('--tau', default=0.2, type=float, help='温度参数，用于 InfoNCE 损失，与 XSimGCL 中的 `tau` 类似')  # 调整温度
    parser.add_argument('--eps', default=0.2, type=float, help='扰动参数，与 XSimGCL 中的 `eps` 类似')
    parser.add_argument('--reg', default=1e-4, type=float, help='隐藏层维度，与 XSimGCL 中的 `hidden` 类似')  # 增加正则化参数
    parser.add_argument('--d_state', default=16, type=int, help='SSM state expansion factor')
    parser.add_argument('--d_conv', default=4, type=int, help='Local convolution width')
    parser.add_argument('--expand', default=2, type=int, help='Block expansion factor')
    return parser.parse_args()
args = ParseArgs()

