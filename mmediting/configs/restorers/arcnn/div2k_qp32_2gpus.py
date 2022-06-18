_base_ = ['./div2k_qp37_2gpus.py']

exp_name = 'arcnn_div2k_qp32'

# dataset settings
data = dict(
    train=dict(dataset=dict(lq_folder='data/div2k/train_lq/qp32')),
    val=dict(lq_folder='data/div2k/valid_lq/qp32'),
    test=dict(lq_folder='data/div2k/valid_lq/qp32'))

# learning policy
total_iters = 300000  # not indicated in the paper
lr_config = dict(periods=[total_iters])

# runtime settings
load_from = './work_dirs/arcnn_div2k_qp37/latest.pth'
