_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_context60.txt',
    logit_scale=50,
    prob_thd=0.5
)

# dataset settings
dataset_type = 'PascalContext60Dataset'
data_root = 'your/path/to/dataset' # TODO 

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))