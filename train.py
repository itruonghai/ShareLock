import sys
import os
import time
import torch

# FeatureUtils requires a staging directory to extract zip shards.
# Default to /tmp/sharelock_staging if TMPDIR is not set.
if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = "/tmp/sharelock_staging"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint

import argparse
from omegaconf import OmegaConf


class ETACallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self._start_time = time.time()
        self._start_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step - self._start_step
        if step < 1:
            return
        elapsed = time.time() - self._start_time
        remaining = trainer.max_steps - trainer.global_step
        eta_min = remaining / (step / elapsed) / 60
        pl_module.log("eta_min", eta_min, prog_bar=True, on_step=True, on_epoch=False)

from sharelock.data.data import DataModule
from sharelock.models.model import ShareLock

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Train ShareLock model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--eval_only", action="store_true", help="Whether to only evaluate the model on the test dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load (.ckpt)")
    args, unknown_args = parser.parse_known_args()

    # Load hyperparameters and checkpoint (if provided)
    config = OmegaConf.load(args.config)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        config = OmegaConf.merge(config, checkpoint["hyper_parameters"])
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)
    
    # Seeding
    pl.seed_everything(config.seed, workers=True)
    
    # Initialize data module
    print("Loading data")
    data_module = DataModule(config)
    
    # Initialize model
    print("Loading model")
    if args.checkpoint is not None:
        model = ShareLock.load_from_checkpoint(args.checkpoint, config=config, weights_only=False)
    else:
        model = ShareLock(config)
    
    # Initialize callbacks
    callbacks = [ETACallback()]
    checkpointing = ModelCheckpoint(
        filename="best_model",
        save_top_k=1,
        monitor="validation_loss",
        mode="min",
    )
    callbacks.append(checkpointing)
    if config.training.early_stopping:
        min_delta = config.training.get("early_stopping_min_delta", 0.001)
        callbacks.append(EarlyStopping(
            monitor="validation_loss",
            patience=config.training.early_stopping_patience,
            min_delta=min_delta,
            mode="min",
        ))
    
    # Set up logging for evaluation and results
    logger = pl.loggers.TensorBoardLogger(save_dir=config.logging.save_dir, name=config.experiment_name)
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Initialize trainer
    print("Loading trainer")
    num_gpus = config.training.get("num_gpus", 1)
    precision = config.training.get("precision", "bf16-mixed")
    # Disable rich progress bar when stdout is redirected (e.g. tee to file)
    # so that tail -f shows clean line-by-line output instead of ANSI escape codes.
    enable_progress_bar = sys.stdout.isatty()
    trainer = pl.Trainer(
        logger=logger,
        max_steps=config.training.max_steps,
        log_every_n_steps=config.logging.log_every_n_steps,
        val_check_interval=config.logging.val_check_interval,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        gradient_clip_val=config.training.max_grad_norm,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        devices=num_gpus,
        strategy="ddp" if num_gpus > 1 else "auto",
        precision=precision,
        enable_progress_bar=enable_progress_bar,
        )
    
    if args.eval_only:
        # Load the best model from the checkpoint
        assert args.checkpoint is not None, "Checkpoint must be provided for evaluation"
    else:
        # Train the model
        trainer.fit(model, data_module)

        # PyTorch >= 2.6 defaults weights_only=True which blocks omegaconf types stored
        # in Lightning checkpoints. This checkpoint is locally generated and trusted.
        model = ShareLock.load_from_checkpoint(
            checkpointing.best_model_path, config=config, weights_only=False
        )
    
    # Evaluate the model (skip if test features are not available)
    skip_test = config.get("skip_test", False)
    if not skip_test:
        trainer.test(model, data_module)

    # Print best checkpoint path so run_experiment.sh can pick it up
    if checkpointing.best_model_path:
        print(f"BEST_CHECKPOINT={checkpointing.best_model_path}")
    
    