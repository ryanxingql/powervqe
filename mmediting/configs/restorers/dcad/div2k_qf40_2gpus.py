_base_ = ['./div2k_qp37_2gpus.py']

exp_name = 'dcad_div2k_qf40'

# dataset settings
data = dict(
    train=dict(
        dataset=dict(lq_folder='data/div2k/train_lq/qf40', ext_lq='.jpg')),
    val=dict(lq_folder='data/div2k/valid_lq/qf40', ext_lq='.jpg'),
    test=dict(lq_folder='data/div2k/valid_lq/qf40', ext_lq='.jpg'))

# learning policy
total_iters = 300000
lr_config = dict(periods=[total_iters])

# runtime settings
work_dir = f'./work_dirs/{exp_name}'
load_from = './work_dirs/dcad_div2k_qf50/latest.pth'
