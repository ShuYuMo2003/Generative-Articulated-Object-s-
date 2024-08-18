import torch, numpy as np
import os, os.path as osp
from pprint import pprint

def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }

def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    # print(pred)
    # print(label)
    return s

def eval_instantiation_distance(gen_name, ref_name, N_states=10, N_pcl=2048):
    results = {}
    rs_fn = f"../logs/test/ID_D_matrix/{gen_name}_{ref_name}_{N_states}_{N_pcl}.npz"
    rr_fn = f"../logs/test/ID_D_matrix/{ref_name}_{ref_name}_{N_states}_{N_pcl}.npz"
    ss_fn = f"../logs/test/ID_D_matrix/{gen_name}_{gen_name}_{N_states}_{N_pcl}.npz"
    M_rs = torch.from_numpy(np.load(rs_fn)["D"])
    M_rr = torch.from_numpy(np.load(rr_fn)["D"])
    M_ss = torch.from_numpy(np.load(ss_fn)["D"])
    ret = lgan_mmd_cov(M_rs.t())
    results.update({
        "%s-ID" % k: v for k, v in ret.items()
    })
    ret = knn(M_rr, M_rs, M_ss, 1, sqrt=False)
    results.update({
        "1-NN-ID-%s" % k: v for k, v in ret.items() if 'acc' in k
    })
    # print(M_rs[:5,:5])
    print(gen_name, ref_name)
    # pprint(results)
    final_results = {
        "1-NN-ID-acc": results["1-NN-ID-acc"],
        "lgan_mmd-ID": results["lgan_mmd-ID"],
        "lgan_cov-ID": results["lgan_cov-ID"],
    }
    # final_results = {
    #     "1-NN-ID-acc": float(results["1-NN-ID-acc"]),
    #     "lgan_mmd-ID": float(results["lgan_mmd-ID"]),
    #     "lgam_cov-ID": float(results["lgan_cov-ID"]),
    # }
    pprint(final_results)
    return

N_states = 10
N_pcl = 2048
gen = "gen"
ref = "gt"
eval_instantiation_distance(gen, ref, N_states, N_pcl)
# gen = "K_8_cate_all_v6.1_5455_retrieval"
# eval_instantiation_distance(gen, ref, N_states, N_pcl)