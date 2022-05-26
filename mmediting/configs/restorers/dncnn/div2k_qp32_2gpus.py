import math

exp_name = 'dncnn_div2k_qp32'

# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='DnCNN',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=15,
        if_bn=True),
    pixel_loss=dict(
        type='MSELoss', loss_weight=1.,
        reduction='mean')  # loss_weight = .5 in the paper; see form. (1)
)
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRFolderDataset'
test_dataset_type = 'SRFolderDataset'
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
    dict(type='PairedRandomCrop', gt_patch_size=128),
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
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(
        samples_per_gpu=16,
        drop_last=True),  # batch size = 128 in the paper; here we use 32
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/div2k/train_lq/qp32',
            gt_folder='data/div2k/train_hq',
            pipeline=train_pipeline,
            scale=1)),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/div2k/valid_lq/qp32',
        gt_folder='data/div2k/valid_hq',
        pipeline=test_pipeline,
        scale=1),
    test=dict(
        type=test_dataset_type,
        lq_folder='data/div2k/valid_lq/qp32',
        gt_folder='data/div2k/valid_hq',
        pipeline=test_pipeline,
        scale=1))

# optimizer
lr_main = 1e-4
optimizers = dict(
    # generator=dict(
    #     type='SGD',
    #     lr=0.1,
    #     weight_decay=0.0001,
    #     momentum=0.9,
    # )  # paper
    generator=dict(
        type='Adam',
        lr=lr_main,
    ))

# learning policy
total_iters = 300000
gamma = math.exp(
    math.log(1e-3) /
    total_iters)  # 1e-1 -> 1e-4: 1e-4 = 1e-1 * gamma^(total_iters)
lr_config = dict(policy='Exp', by_epoch=False, gamma=gamma)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = './work_dirs/dncnn_div2k_qp37/latest.pth'
resume_from = None
workflow = [('train', 1)]
