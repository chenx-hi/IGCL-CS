import random
import time

import numpy as np

from model_ogbn import IGCL
from logreg import LogReg
from params import set_params
from util.dataset import load_ogbn_data
from util.data_utils import eval_acc
from util.eval import *
from torch_geometric.utils import spmm

def fix_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = set_params()
fix_seed(args.seed)
print(args)

if torch.cuda.is_available() and args.gpu != -1:
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

data = load_ogbn_data(args).to_train(device)
h = torch.spmm(data.partition.T, data.x)
bce_loss = torch.nn.BCEWithLogitsLoss()

accs = []
times = []

model = IGCL(data.num_features, args.hidden_channels, args.mlp_layers, args.proj_layers, args.tau,
                  args.beta, args.lamda, args.batch_norm, args.layer_norm).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
model.train()
model.encoder_target.eval()

time_start = time.time()
for epoch in range(int(args.epochs) + 1):
    optimizer.zero_grad()
    loss = model(data.x, h, data.partition, data.coarse_g)
    loss.backward()
    optimizer.step()
    model.update_moving_average()
    print(loss)
time_end = time.time()
times.append(time_end - time_start)
mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)

model.eval()
data = data.to_test(device)

for i in range(5):
    best_acc_val = 0
    best_loss_val = 1e9
    final_test = 0
    with torch.no_grad():
        emb = model.get_emb(data.x, data.graph, args.k_hop)
    log = LogReg(args.hidden_channels, data.num_classes).to(device)
    opt = torch.optim.Adam(log.parameters(), lr=args.cls_lr, weight_decay=args.cls_weight_decay)
    for _ in range(args.cls_epochs):
        log.train()
        opt.zero_grad()
        prob_train = torch.nn.functional.log_softmax(log(emb[data.idx_train]), dim=1)
        loss_cls = torch.nn.functional.nll_loss(prob_train, data.y[data.idx_train])
        loss_cls.backward()
        opt.step()

        log.eval()
        prob = torch.nn.functional.log_softmax(log(emb), dim=1)
        loss_val = torch.nn.functional.nll_loss(prob[data.idx_valid], data.y[data.idx_valid])
        acc_val = eval_acc(prob[data.idx_valid], data.y[data.idx_valid])
        acc_test = eval_acc(prob[data.idx_test], data.y[data.idx_test])

        if acc_val >= best_acc_val and best_loss_val >= loss_val:
            #print("better classification!")
            best_acc_val = max(acc_val, best_acc_val)
            best_loss_val = loss_val
            final_test = max(acc_test, final_test)
        #nni.report_intermediate_result(final_test.item())

    accs.append(final_test.item())
    print(final_test)
print(np.mean(accs))
print(np.mean(times))
print(mem_gb)
