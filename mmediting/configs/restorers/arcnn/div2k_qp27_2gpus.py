exp_name = 'arcnn_div2k_qp27'

# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='ARCNN',
        in_channels=3,
        mid_channels_1=64,
        mid_channels_2=32,
        mid_channels_3=16,
        out_channels=3,
        in_kernel_size=9,
        mid_kernel_size_1=7,
        mid_kernel_size_2=1,
        out_kernel_size=5),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'
    ),  # keep the color type. ref: https://mmcv.readthedocs.io/en/latest/api.html#mmcv.image.imread
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=128),  # 38 in the paper
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
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
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
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16,
                          drop_last=True),  # 128 in the paper; here we use 32
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/div2k/train_lq/qp27',
            gt_folder='data/div2k/train_hq',
            pipeline=train_pipeline,
            scale=1)),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/div2k/valid_lq/qp27',
        gt_folder='data/div2k/valid_hq',
        pipeline=test_pipeline,
        scale=1),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/div2k/valid_lq/qp27',
        gt_folder='data/div2k/valid_hq',
        pipeline=test_pipeline,
        scale=1))

# optimizer
lr_main = 1e-4  # 1 in the paper
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=lr_main,
        #paramwise_cfg=dict(
        #    custom_keys={'conv_after_body': dict(lr_mult=0.1)},
        #    bias_lr_mult=0.1,
        #)
    ))

# learning policy
total_iters = 300000  # not indicated in the paper
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[total_iters],
    restart_weights=[1],
    min_lr=lr_main / 1e3)  # not indicated in the paper

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = './work_dirs/arcnn_div2k_qp37/latest.pth'
resume_from = None
workflow = [('train', 1)]
