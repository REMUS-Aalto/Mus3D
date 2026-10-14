"""
Shared training launcher: Hydra + Accelerate boilerplate behind one call so the
per-method entry points (the *_train_launcher.py files) stay thin.
"""

import os
import sys
sys.dont_write_bytecode = True

from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ema_pytorch import EMA

from utils.torch_common import get_world_size, count_parameters, set_seed


def run(config_name, TrainerClass, params_fn=None, report_fn=None, use_ema=True,
        ema_fn=None):
    """
    config_name: Hydra config (e.g. 'default_surf.yaml').
    TrainerClass: Trainer subclass to drive training.
    params_fn:   model -> params to optimize (default: all model params).
    report_fn:   model -> module whose param count is logged (default: model).
    use_ema:     keep an eval-time EMA of the model (off for SURF).
    ema_fn:      model -> module to keep an EMA OF (default: the whole model).
                 Models carrying frozen encoders must narrow this to the
                 trainable part: EMA deepcopies its target, and weight_norm'd
                 modules (e.g. the SAME autoencoder) are not deepcopyable.
    """
    params_fn = params_fn or (lambda m: m.parameters())
    report_fn = report_fn or (lambda m: m)
    ema_fn = ema_fn or (lambda m: m)

    # Absolute: `main` lives in an imported module, so Hydra would otherwise
    # resolve a relative path as the (unimportable) package `configs`.
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def main(cfg: DictConfig):
        # Resume: merge command-line overrides onto the saved config.
        if cfg.trainer.ckpt_dir is not None:
            overrides = [e for e in HydraConfig.get().overrides.task if isinstance(e, str)]
            cfg_ckpt = OmegaConf.load(f'{cfg.trainer.ckpt_dir}/config.yaml')

            # Group overrides (`data=foo`, `model=bar`) select a config file
            # and are already baked into the snapshot; dropped here since
            # `from_dotlist` would otherwise overwrite the whole node with a
            # literal string.
            kept = []
            for e in overrides:
                key = e.split("=", 1)[0].lstrip("+~")
                node = OmegaConf.select(cfg_ckpt, key)
                if OmegaConf.is_config(node):
                    print(f"[resume] ignoring group override '{e}' "
                          f"-- '{key}' comes from the checkpoint's own config")
                    continue
                # `from_dotlist` has no notion of hydra's +/~ prefixes; strip
                # them so it merges onto the snapshot instead of adding a
                # literal "+trainer" key.
                kept.append(f"{key}={e.split('=', 1)[1]}" if e[0] in "+~" else e)

            override_conf = OmegaConf.from_dotlist(kept)
            cfg_merged = OmegaConf.merge(cfg_ckpt, override_conf)
        else:
            cfg_merged = cfg
        cfg_ = cfg_merged

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accel = Accelerator(
            mixed_precision=cfg_.trainer.amp,
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            project_config=ProjectConfiguration(project_dir=cfg_.trainer.output_dir),
            kwargs_handlers=[ddp_kwargs],
            log_with='wandb',
        )
        accel.init_trackers(cfg_.trainer.logger.project_name, config=OmegaConf.to_container(cfg_),
                            init_kwargs={"wandb": {"name": cfg_.trainer.logger.run_name,
                                                   "dir": cfg_.trainer.output_dir}})
        if accel.is_main_process:
            print(f"->->-> DDP Initialized. World size: {get_world_size()}")
        set_seed(cfg_.trainer.seed)

        train_dataset = hydra.utils.instantiate(cfg_.data.train)
        # Datasets with variable-shape items expose their own collate_fn,
        # since default_collate cannot stack those. `prefetch_factor` is set
        # explicitly to bound shared-memory use across workers.
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg_.trainer.batch_size,
            num_workers=cfg_.trainer.num_workers, pin_memory=True,
            collate_fn=getattr(train_dataset, "collate_fn", None),
            persistent_workers=(cfg_.trainer.num_workers > 0),
            **({"prefetch_factor": cfg_.trainer.prefetch_factor}
               if cfg_.trainer.num_workers > 0
               and "prefetch_factor" in cfg_.trainer else {}))

        # Held-out demo/eval source; None unless `data.val` exists.
        val_dataset = (
            hydra.utils.instantiate(cfg_.data.val) if "val" in cfg_.data else None
        )

        model = hydra.utils.instantiate(cfg_.model)
        model = model.float()  # avoid MPS float64 issues
        model.train()

        # `trainer.use_ema` can turn the shadow off from the config on top of
        # the caller's default; the shadow is a full fp32 copy of the model
        # and only affects eval-time averaging, not the gradient.
        want_ema = use_ema and cfg_.trainer.get("use_ema", True)
        if use_ema and not want_ema:
            if accel.is_main_process:
                print("[launch] trainer.use_ema=false -- no EMA shadow this run. "
                      "`metrics_for_best_ckpt` and eval then use the LIVE weights.")

        ema = None
        if want_ema and accel.is_main_process:
            ema = EMA(ema_fn(model), **cfg_.trainer.ema)
            ema.float()
            ema.to(accel.device)

        opt = hydra.utils.instantiate(cfg_.optimizer.optimizer)(params=params_fn(model))
        sche = hydra.utils.instantiate(cfg_.optimizer.scheduler)(optimizer=opt)

        if accel.is_main_process:
            print(f"=== Params: {count_parameters(report_fn(model)) / 1e6:.2f} M "
                  f"| batch {cfg_.trainer.batch_size} | chunks {len(train_dataset)} ===")

        train_dataloader, model, opt, sche = accel.prepare(train_dataloader, model, opt, sche)

        TrainerClass(model, ema, opt, sche, train_dataloader, accel, cfg_,
                     ckpt_dir=cfg_.trainer.ckpt_dir,
                     val_dataset=val_dataset).start_training()

    main()
