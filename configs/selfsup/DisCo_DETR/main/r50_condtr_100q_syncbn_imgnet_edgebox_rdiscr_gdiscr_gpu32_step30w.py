_base_ = '../imgnet_edgebox_crop.py'

# ------- hyper parameters -------

runner_type = 'epoch'
steps = 60
step_drop = 40

by_epoch = True

backbone_channels = 2048
hidden_dim = 256
num_patches = 10
aux_loss = True
num_dec_layers = 6

# ------- ablation parameters -------
dist_use = False
contrast_use = True
feature_recon = True
all_layers = False
match_layer1 = False
part_contrast = False
batch_contrast = False
positive_from_cosine = False
contrast_transform = True
part_contrast_transform = False
part_contrast_transform_proj = False
only_positive = True
tgt_use=False
with_patch=True
my_transform=False


query_save=True

# -------- model --------

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],
    frozen_stages=4,
    norm_cfg=dict(type='FrozenBN'))  # add pe in up-detr

transformer = dict(
    type='ConditionalTR',
    hidden_dim=hidden_dim,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=num_dec_layers,
    dim_feedforward=2048,
    dropout=0.1,
    normalize_before=False,
    return_intermediate_dec=aux_loss,
    dist_use=dist_use,
    pattern=0,
    match_layer1=match_layer1,
    part_contrast=part_contrast,
    tgt_use=tgt_use,
    with_patch=with_patch,
    num_queries=100,)

tr_class = transformer['type']

pred_head = dict(
    type='SiameseDETRPredictHead',
    aux_loss=aux_loss,
    hidden_dim=hidden_dim,
    size_average=True,
    feature_recon=True,
    backbone_channels=backbone_channels,
    contrast_use=contrast_use,
    all_layers=all_layers,
    part_contrast=part_contrast,
    batch_contrast=batch_contrast,
    positive_from_cosine=positive_from_cosine,
    contrast_transform=contrast_transform,
    part_contrast_transform=part_contrast_transform,
    only_positive = only_positive,
    part_contrast_transform_proj = part_contrast_transform_proj
)

    # matcher_cfg=dict(
    #     cost_class=1,
    #     cost_bbox=5,
    #     cost_giou=2)

encoder_head = dict(
    type='SiameseDETREncoderGlobalLatentHead',
    projector=dict(
        type='NonLinearNeckV3',
        in_channels=256,
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

position_embedding = dict(type='sine', hidden_dim=hidden_dim)

model = dict(
    type='DisCoDETR',  # pe + query
    pretrained='bakcbone_path',
    freeze_backbone=True,
    query_shuffle=False,
    num_queries=100,
    num_patches=num_patches,
    box_disturbance=0.1,
    backbone_channels=backbone_channels,
    feature_recon=True,
    hidden_dim=hidden_dim,
    weight_dict=dict(
        loss_enc_global_contra=10,
        loss_ce=1,
        loss_bbox=5,
        loss_giou=2,
        loss_feature=5,
        loss_contrast=2,
        num_repeat=num_dec_layers),
    backbone=backbone,
    position_embedding=position_embedding,
    transformer=transformer,
    pred_head=pred_head,
    encoder_head=encoder_head)

# ------- others -------

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4)

update_interval=1

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
