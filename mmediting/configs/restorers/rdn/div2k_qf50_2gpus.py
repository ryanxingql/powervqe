_base_ = ['./div2k_qp37_2gpus.py']

exp_name = 'rdn_div2k_qf50'

# dataset settings
data = dict(
    train=dict(
        dataset=dict(lq_folder='data/div2k/train_lq/qf50', ext_lq='.jpg')),
    val=dict(lq_folder='data/div2k/valid_lq/qf50', ext_lq='.jpg'),
    test=dict(lq_folder='data/div2k/valid_lq/qf50', ext_lq='.jpg'))
