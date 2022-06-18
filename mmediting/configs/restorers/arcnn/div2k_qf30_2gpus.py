_base_ = ['./div2k_qp37_2gpus.py']

exp_name = 'arcnn_div2k_qf30'

# dataset settings
data = dict(
    train=dict(
        dataset=dict(lq_folder='data/div2k/train_lq/qf30', ext_lq='.jpg')),
    val=dict(lq_folder='data/div2k/valid_lq/qf30', ext_lq='.jpg'),
    test=dict(lq_folder='data/div2k/valid_lq/qf30', ext_lq='.jpg'))

# learning policy
total_iters = 300000  # not indicated in the paper
lr_config = dict(periods=[total_iters])

# runtime settings
load_from = './work_dirs/arcnn_div2k_qf50/latest.pth'
