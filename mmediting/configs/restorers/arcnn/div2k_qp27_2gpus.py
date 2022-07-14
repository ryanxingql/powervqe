_base_ = ['./div2k_qp37_2gpus.py']

exp_name = 'arcnn_div2k_qp27'

# dataset settings
data = dict(
    train=dict(dataset=dict(lq_folder='data/div2k/train_lq/qp27')),
    val=dict(lq_folder='data/div2k/valid_lq/qp27'),
    test=dict(lq_folder='data/div2k/valid_lq/qp27'))

# learning policy
total_iters = 300000
lr_config = dict(periods=[total_iters])

# runtime settings
work_dir = f'./work_dirs/{exp_name}'
load_from = './work_dirs/arcnn_div2k_qp37/latest.pth'
