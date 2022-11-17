_base_ = '../yolox/yolox_s_8xb8-300e_coco.py'
fp16 = dict(loss_scale='dynamic')

# model settings
model = dict(neck=dict(use_evcblock=True))

auto_scale_lr = dict(enable=True, base_batch_size=64)
