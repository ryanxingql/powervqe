exp_name = 'stdf_ldv_v2'

# model settings
model = dict(
    type='STDF',
    generator=dict(
        type='STDFNet',
        in_channels=3,
        out_channels=3,
        radius=3,
        nf_stdf=32,
        nb_stdf=3,
        deform_ks=3,
        nf_stdf_out=64,
        nf_qe=48,
        nb_qe=6,
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_pipeline = [
    dict(
        type='GenerateFrameIndicesEDVR',
        interval_list=[1],
        filename_tmpl='f{:03d}',
        idx_start_from=1),
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
        type='GenerateFrameIndiceswithPaddingEDVR',
        padding='reflection_circle',
        idx_start_from=1),
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

train_dataset_type = 'SRLDVDataset'
val_dataset_type = 'SRLDVDataset'

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=8, drop_last=True),  # 32 in total
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=300,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/ldv_v2/train_lq',
            gt_folder='data/ldv_v2/train_gt',
            num_input_frames=7,
            pipeline=train_pipeline,
            scale=1,
        )),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/ldv_v2/valid_lq',
        gt_folder='data/ldv_v2/valid_gt',
        num_input_frames=7,
        pipeline=test_pipeline,
        scale=1,
        # test only 100 frames each video to decrease the test time
        max_need_frms=100,
        test_mode=True,  # turn on max_need_frms
    ),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/ldv_v2/test_lq',  # 'data/mfqe_v2/test_lq'
        gt_folder='data/ldv_v2/test_gt',  # 'data/mfqe_v2/test_gt'
        num_input_frames=7,
        pipeline=test_pipeline,
        scale=1,
    ),
)

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999), eps=1e-8))

# learning policy
total_iters = 1000000
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
find_unused_parameters = False
workflow = [('train', 1)]
