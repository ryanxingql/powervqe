_base_ = ['./div2k_qp37_lmdb_2gpus.py']

exp_name = 'cbdnet_div2k_qf50'

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='lmdb',
        db_path='data/div2k/train_lq_sub_qf50.lmdb',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='lmdb',
        db_path='data/div2k/train_hq_sub.lmdb',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    # dict(type='PairedRandomCrop', gt_patch_size=128),  # no need
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='lmdb',
        db_path='data/div2k/valid_lq_qf50.lmdb',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='lmdb',
        db_path='data/div2k/valid_hq.lmdb',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'lq_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    train=dict(
        dataset=dict(
            lq_folder='data/div2k/train_lq_sub_qf50.lmdb',
            pipeline=train_pipeline,
        )),
    val=dict(
        lq_folder='data/div2k/valid_lq_qf50.lmdb',
        pipeline=test_pipeline,
    ),
    test=dict(
        lq_folder='data/div2k/valid_lq_qf50.lmdb',
        pipeline=test_pipeline,
    ))

# runtime settings
work_dir = f'./work_dirs/{exp_name}'
