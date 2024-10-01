import torch
import numpy as np
import torch.nn.functional as F


def cal_mae_acc_rank(logits, is_sto=True):
    if is_sto:
        r_dim, s_dim, out_dim = logits.shape
        assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
        logits = logits.view(r_dim, s_dim, int(out_dim / 2), 2)
        logits = torch.argmax(logits, dim=-1)
        logits = torch.sum(logits, dim=-1)
        logits = torch.mean(logits.float(), dim=0)
        logits = logits.cpu().data.numpy()
        pred_label = np.rint(logits)
        # targets = targets.cpu().data.numpy()
        # mae = sum(abs(logits - targets)) * 1.0 / len(targets)
        # acc = sum(np.rint(logits) == targets) * 1.0 / len(targets)
    else:
        s_dim, out_dim = logits.shape
        assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
        logits = logits.view(s_dim, int(out_dim / 2), 2)
        logits = torch.argmax(logits, dim=-1)
        logits = torch.sum(logits, dim=-1)
        logits = logits.cpu().data.numpy()
        pred_label = np.rint(logits)
        # targets = targets.cpu().data.numpy()
        # mae = sum(abs(logits - targets)) * 1.0 / len(targets)
        # acc = sum(np.rint(logits) == targets) * 1.0 / len(targets)
    return pred_label



def cal_mae_acc_reg(logits, is_sto=True):
    if is_sto:
        logits = logits.mean(dim=0)

    # assert logits.view(-1).shape == targets.shape, "logits {}, targets {}".format(
    #     logits.shape, targets.shape)

    logits = logits.cpu().data.numpy().reshape(-1)
    pred_label = np.rint(logits)
    # targets = targets.cpu().data.numpy()
    # mae = sum(abs(logits - targets)) * 1.0 / len(targets)
    # acc = sum(np.rint(logits) == targets) * 1.0 / len(targets)

    return pred_label


def cal_mae_acc_cls(logits, is_sto=True):
    if is_sto:
        r_dim, s_dim, out_dim = logits.shape
        label_arr = torch.arange(0, out_dim).float().cuda()
        probs = F.softmax(logits, -1)
        # exp = torch.sum(probs * label_arr, dim=-1)
        # exp = torch.mean(exp, dim=0)
        max_a = torch.mean(probs, dim=0)
        max_data = max_a.cpu().data.numpy()
        max_data = np.argmax(max_data, axis=1)
        # exp_data = exp.cpu().data.numpy()
        pred_label = np.rint(max_data)
        
        # mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)
        # acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

    else:
        s_dim, out_dim = logits.shape
        probs = F.softmax(logits, -1)
        probs_data = probs.cpu().data.numpy()
        max_data = np.argmax(probs_data, axis=1)
        label_arr = np.array(range(out_dim))
        exp_data = np.sum(probs_data * label_arr, axis=1)
        pred_label = np.rint(exp_data)
        
        # mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)
        # acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

    return pred_label


def get_label_method(main_loss_type):
    assert main_loss_type in ['cls', 'reg', 'rank'], \
        "main_loss_type not in ['cls', 'reg', 'rank'], loss type {%s}" % (
            main_loss_type)
    if main_loss_type == 'cls':
        return cal_mae_acc_cls
    elif main_loss_type == 'reg':
        return cal_mae_acc_reg
    elif main_loss_type == 'rank':
        return cal_mae_acc_rank
    else:
        raise AttributeError('main loss type: {}'.format(main_loss_type))
