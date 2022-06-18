_base_ = ['./div2k_qp37_2gpus.py']

exp_name = 'dncnn_div2k_qp42'

# dataset settings
data = dict(
    train=dict(dataset=dict(lq_folder='data/div2k/train_lq/qp42')),
    val=dict(lq_folder='data/div2k/valid_lq/qp42'),
    test=dict(lq_folder='data/div2k/valid_lq/qp42'))

# learning policy
total_iters = 300000
lr_config = dict(periods=[total_iters])

# runtime settings
load_from = './work_dirs/dncnn_div2k_qp37/latest.pth'
