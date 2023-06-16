from torcheval.metrics import MulticlassConfusionMatrix

def get_ious(preds, labels, n_class, device):

    mcm = MulticlassConfusionMatrix(n_class).to(device)
    mcm.update(preds, labels)
    mcm_results = mcm.compute()
    tps = mcm_results.diagonal()
    fps = mcm_results.sum(axis=1) - tps
    fns = mcm_results.sum(axis=0) - tps

    return tps, fps, fns
