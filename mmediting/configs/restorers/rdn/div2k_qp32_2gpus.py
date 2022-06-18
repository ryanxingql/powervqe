_base_ = ['./div2k_qp37_2gpus.py']

exp_name = 'rdn_div2k_qp32'

# dataset settings
data = dict(
    train=dict(dataset=dict(lq_folder='data/div2k/train_lq/qp32')),
    val=dict(lq_folder='data/div2k/valid_lq/qp32'),
    test=dict(lq_folder='data/div2k/valid_lq/qp32'))

# learning policy
total_iters = 500000
lr_config = dict(step=[100000, 200000, 300000, 400000])

# runtime settings
load_from = './work_dirs/rdn_div2k_qp37/latest.pth'
