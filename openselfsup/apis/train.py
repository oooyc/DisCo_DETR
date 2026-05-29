
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, DistSamplerSeedHook, EpochBasedRunner, \
    IterBasedRunner
import os 
from openselfsup.datasets import build_dataloader
from openselfsup.hooks import build_hook, DistOptimizerHook
from openselfsup.utils import get_root_logger, print_log, build_optimizer, parse_losses

def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data, mode=train_mode)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs

def train_model(model,
                dataset,
                cfg,
                distributed=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(
            model, dataset, cfg, logger=logger, timestamp=timestamp, meta=meta)
    else:
        _non_dist_train(
            model, dataset, cfg, logger=logger, timestamp=timestamp, meta=meta)

def _dist_train(model, dataset, cfg, logger=None, timestamp=None, meta=None):
    runner_type = cfg.get('runner_type', 'epoch')
    assert runner_type in ['epoch', 'iter'], f'runner type {runner_type} not supported'
    print_log(f'create {runner_type}-based runner', 'root')
    
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    sampler_type = ('DistributedGivenIterationSampler', cfg.max_iters) if runner_type == 'iter' else None
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True,
            shuffle=True,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False),
            prefetch=cfg.prefetch,
            img_norm_cfg=cfg.img_norm_cfg,
            collate_fn=cfg.data.get('collate_fn', None),
            sampler_type=sampler_type,) for ds in dataset
    ]
    optimizer = build_optimizer(model, cfg.optimizer)
    if 'use_fp16' in cfg and cfg.use_fp16:
        try:
            import apex
            model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level="O1")
            print_log('**** Initializing mixed precision done. ****')
        except ImportError:
            rank, world_size = get_dist_info()
            if rank == 0:
                print('apex is not installed')
    # print(torch.cuda.current_device())
    # put model on gpus
    # device_ids=[torch.cuda.current_device()],
    model = MMDistributedDataParallel(
        model if next(model.parameters()).is_cuda else model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    # def create_hook(param_name):
    #     def hook_fn(grad):
    #         if grad is None:
    #             print(f"Parameter {param_name} received no gradient")
    #         elif grad.sum() == 0:
    #             print(f"Parameter {param_name} received zero gradient")
    #     return hook_fn

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         param.register_hook(create_hook(name))
    #     else:
    #         print(f"Parameter {name} does not require gradient")
    # build runner
    if runner_type == 'epoch':
        runner = EpochBasedRunner(
            model,
            batch_processor if not hasattr(model, 'train_step') else None,
            optimizer,
            cfg.work_dir,
            logger=logger,
            meta=meta,
            update_interval=cfg.update_interval,
            query_save=cfg.query_save,
            tr_class=cfg.tr_class)
        runner.register_hook(DistSamplerSeedHook())
    else:
        runner = IterBasedRunner(
            model,
            None,
            optimizer,
            cfg.work_dir,
            logger=logger,
            meta=meta)

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register hooks
    # optimizer_config = dict(grad_clip=dict(max_norm=0.1))
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    # checkpoint_config = dict(interval=5000, by_epoch=False)
    # lr_config = dict(policy='Step', step=step_drop, gamma=0.1, by_epoch=False)
    # log_config = dict(
    #     interval=50,
    #     hooks=[
    #         dict(type='TextLoggerHook', by_epoch=False),
    #         dict(type='TensorboardLoggerHook', by_epoch=False)
    #     ])
    # register custom hooks
    for hook in cfg.get('custom_hooks', ()):
        if hook.type == 'DeepClusterHook':
            common_params = dict(dist_mode=True, data_loaders=data_loaders)
        else:
            common_params = dict(dist_mode=True)
        runner.register_hook(build_hook(hook, common_params))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
        for loader in data_loaders:
            if hasattr(loader.sampler, 'set_last_iter'):
                loader.sampler.set_last_iter(runner.iter)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    schedule = cfg.total_epochs if runner_type == 'epoch' else cfg.max_iters
    runner.run(data_loaders, cfg.workflow, schedule)

def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None):

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            shuffle=True,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False),
            prefetch=cfg.prefetch,
            img_norm_cfg=cfg.img_norm_cfg,
            collate_fn=cfg.data.get('collate_fn', None),) for ds in dataset
    ]

    if 'use_fp16' in cfg and cfg.use_fp16 == True:
        raise NotImplementedError('apex do not support non_dist_train!')
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    optimizer = build_optimizer(model, cfg.optimizer)

    # build runner
    runner_type = cfg.get('runner_type', 'epoch')
    assert runner_type in ['epoch', 'iter'], f'runner type {runner_type} not support'
    print_log(f'create {runner_type}-based runner', 'root')
    if runner_type == 'epoch':
        runner = EpochBasedRunner(
            model,
            batch_processor if not hasattr(model, 'train_step') else None,
            optimizer,
            cfg.work_dir,
            logger=logger,
            meta=meta)
    else:
        runner = IterBasedRunner(
            model,
            None,
            optimizer,
            cfg.work_dir,
            logger=logger,
            meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # register custom hooks
    for hook in cfg.get('custom_hooks', ()):
        if hook.type == 'DeepClusterHook':
            common_params = dict(dist_mode=False, data_loaders=data_loaders)
        else:
            common_params = dict(dist_mode=False)
        runner.register_hook(build_hook(hook, common_params))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    schedule = cfg.total_epoch if runner_type == 'epoch' else cfg.max_iters
    print('start training')
    runner.run(data_loaders, cfg.workflow, schedule)


