_base_ = ['./div2k_qp37_lmdb_2gpus.py']

exp_name = 'cbdnet_div2k_qf20'

# dataset settings
data = dict(
    train=dict(dataset=dict(lq_folder='data/div2k/train_lq_sub_qf20.lmdb')),
    val=dict(lq_folder='data/div2k/valid_lq_qf20.lmdb'),
    test=dict(lq_folder='data/div2k/valid_lq_qf20.lmdb'))

# learning policy
total_iters = 300000
lr_config = dict(periods=[total_iters])

# runtime settings
work_dir = f'./work_dirs/{exp_name}'
load_from = './work_dirs/cbdnet_div2k_qf50/latest.pth'
