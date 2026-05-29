_base_ = '../imgnet_edgebox_crop.py'

# ------- hyper parameters -------

runner_type = 'epoch'
steps = 60
step_drop = 40

by_epoch = True

multi_scale_features_backbone_strides = (8, 16, 32)
multi_scale_features_backbone_num_channels = (512, 1024, 2048)
backbone_channels = sum(multi_scale_features_backbone_num_channels)
num_queries = 300

hidden_dim = 256
num_patches = 10
aux_loss = True
num_dec_layers = 6
num_classes = 2
# ------- ablation parameters -------
dist_use = False
feature_recon = True
contrast_transform = True


query_save=True
encoder_head = dict(
    type='SiameseDETREncoderGlobalLatentHead',
    projector=dict(
        type='NonLinearNeckV3',
        in_channels=256*len(multi_scale_features_backbone_strides),
        hid_channels=256,
        out_channels=256,
        sync_bn=True,
        with_avg_pool=True),
    predictor=dict(
        type='NonLinearNeckV2',
        in_channels=256,
        hid_channels=256,
        out_channels=256,
        sync_bn=True,
        with_avg_pool=False),
    size_average=True)

# encoder_head = None

# -------- model --------

# backbone = dict(
#     type='ResNet',
#     depth=50,
#     in_channels=3,
#     out_indices=[2, 3, 4],
#     frozen_stages=4,
#     norm_cfg=dict(type='FrozenBN'))  # add pe in up-detr

backbone = dict(
    type='PResNet',
    depth=50,
    variant='d',
    freeze_at=0,
    return_idx=[1, 2, 3],
    num_stages=4,
    freeze_norm=True,
    pretrained=True, 
    pretrained_path=None,)  # add pe in up-detr

# PResNet:
#   depth: 50
#   variant: d
#   freeze_at: 0
#   return_idx: [1, 2, 3]
#   num_stages: 4
#   freeze_norm: True
#   pretrained: True 

transformer = dict(
    type='RTDETR',
    num_classes = num_classes,
    num_queries = num_queries,
    #--------------------encoder------------------------
    in_channels=[512, 1024, 2048],
    feat_strides=[8, 16, 32],

    # intra
    hidden_dim=256,
    use_encoder_idx=[2],
    num_encoder_layers=1,
    nhead=8,
    dim_feedforward=1024,
    dropout=0.,
    enc_act='gelu',
    pe_temperature=10000,
    
    # cross
    expansion=1.0,
    depth_mult=1,
    act='silu',

    # eval
    eval_spatial_size=[640, 640],
    #--------------------encoder------------------------

    #--------------------decoder------------------------
    feat_channels=[256, 256, 256],

    num_levels=3,

    num_decoder_layer= 6,
    num_denoising=100,
    
    eval_idx=-1,
    eval_spatial_size=[640, 640],
    #--------------------decoder------------------------

    multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],
)

tr_class = transformer['type']

pred_head = dict(
    type='SiameseRTDETRPredictHead',
    aux_loss=aux_loss,
    hidden_dim=hidden_dim,
    size_average=True,
    feature_recon=feature_recon,
    backbone_channels=backbone_channels,
    num_classes=num_classes,
)
    # matcher_cfg=dict(
    #     cost_class=1,
    #     cost_bbox=5,
    #     cost_giou=2))

decoder_head = dict(
    type='SiameseRTDETRDecoderLocalLatentHead',
    hidden_dim=hidden_dim,
)

# position_embedding = dict(type='sine', hidden_dim=hidden_dim)

model = dict(
    type='disCo_rt_detr',  # pe + query
    # pretrained=None,
    pretrained='bakcbone_path',
    freeze_backbone=True,
    query_shuffle=False,
    num_queries=300,
    num_patches=num_patches,
    box_disturbance=0.1,
    backbone_channels=backbone_channels,
    feature_recon=feature_recon,
    hidden_dim=hidden_dim,
    weight_dict=dict(
        loss_enc_global_contra=10,
        # loss_ce=1,
        loss_vf1=1,
        loss_bbox=5,
        loss_giou=2,
        loss_feature=3,
        loss_contrast=2,
        num_repeat=num_dec_layers,),
    backbone=backbone,
    transformer=transformer,
    pred_head=pred_head,
    encoder_head=encoder_head,
    multi_scale_features=True,
    multi_scale_features_backbone_strides=multi_scale_features_backbone_strides,
    multi_scale_features_backbone_num_channels=multi_scale_features_backbone_num_channels,
    num_encoder_layers = 1,
    use_encoder_idx = [2],
    eval_spatial_size = [640, 640],
    pe_temperature=10000,
)

# ------- others -------

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4)

update_interval=1
new_siamese=True

checkpoint_config = dict(interval=update_interval, max_keep_ckpts=2, by_epoch=by_epoch)
lr_config = dict(policy='Step', step=step_drop, gamma=0.1, by_epoch=by_epoch)
log_config = dict(
    interval=update_interval,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=by_epoch),
        dict(type='TensorboardLoggerHook', by_epoch=by_epoch)
    ])

max_iters = steps
total_epochs = steps
