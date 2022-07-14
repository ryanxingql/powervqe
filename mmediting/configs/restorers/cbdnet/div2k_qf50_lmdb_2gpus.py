_base_ = ['./div2k_qp37_lmdb_2gpus.py']

exp_name = 'cbdnet_div2k_qf50'

# dataset settings
data = dict(
    train=dict(dataset=dict(lq_folder='data/div2k/train_lq_sub_qf50.lmdb')),
    val=dict(lq_folder='data/div2k/valid_lq_qf50.lmdb'),
    test=dict(lq_folder='data/div2k/valid_lq_qf50.lmdb'))

# runtime settings
work_dir = f'./work_dirs/{exp_name}'
