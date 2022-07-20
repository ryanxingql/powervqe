exp_name = 'mfqev2_ldv_v2_non_pqf'

# model settings
model = dict(
    type='MFQEv2Restorer',
    generator=dict(
        type='MFQEv2',
        in_channels=3,
        out_channels=3,
        nf=32,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
total_iters = 600000
train_cfg = dict(fix_spynet_iter=total_iters // 10)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_pipeline = [
    dict(
        type='GenerateFrameIndicesMFQE',
        filename_tmpl='f{:03d}',
    ),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
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
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

test_pipeline = [
    dict(
        type='GenerateFrameIndicesMFQE',
        filename_tmpl='f{:03d}',
    ),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
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
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

train_dataset_type = 'LDPNonPQFDataset'
val_dataset_type = 'LDPNonPQFDataset'

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=32, drop_last=True),  # 128 in total
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=300,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/ldv_v2/train_lq',
            gt_folder='data/ldv_v2/train_gt',
            i_frame_idx=1,
            pipeline=train_pipeline,
            scale=1,
        )),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/ldv_v2/valid_lq',
        gt_folder='data/ldv_v2/valid_gt',
        i_frame_idx=1,
        pipeline=test_pipeline,
        scale=1,
        max_need_frms=
        100,  # test only 100 frames each video; or the val time is so long
        test_mode=True,  # turn on max_need_frms
    ),
    test=dict(
        type=val_dataset_type,
        lq_folder=
        'data/ldv_v2/test_lq',  # 'data/ldv_v2/test_lq' or 'data/mfqe_v2/test_lq'
        gt_folder=
        'data/ldv_v2/test_gt',  # 'data/ldv_v2/test_gt' or 'data/mfqe_v2/test_gt'
        i_frame_idx=1,
        pipeline=test_pipeline,
        scale=1,
    ),
)

# optimizer
lr_main = 1e-4
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=lr_main,
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# learning policy
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[total_iters],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non-distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
find_unused_parameters = True
workflow = [('train', 1)]
