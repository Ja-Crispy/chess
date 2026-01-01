"""
Base trainer class for chess models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm
import wandb


class BaseTrainer:
    """
    Base trainer for chess models.

    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging (wandb)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        use_wandb: bool = True,
        wandb_project: str = 'chess-models'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                pieces=batch.get('pieces'),
                squares=batch.get('squares'),
                mask=batch.get('mask'),
                features=batch.get('features'),  # For NNUE
                legal_mask=batch.get('legal_mask')
            )

            # Compute loss
            losses = self.loss_fn(outputs, batch)

            # Backward pass
            self.optimizer.zero_grad()
            losses['loss'].backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += losses['loss'].item()
            total_policy_loss += losses.get('policy_loss', 0).item() if 'policy_loss' in losses else 0
            total_value_loss += losses.get('value_loss', 0).item() if 'value_loss' in losses else 0
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['loss'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/loss': losses['loss'].item(),
                    'train/policy_loss': losses.get('policy_loss', 0).item() if 'policy_loss' in losses else 0,
                    'train/value_loss': losses.get('value_loss', 0).item() if 'value_loss' in losses else 0,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

            self.global_step += 1

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                pieces=batch.get('pieces'),
                squares=batch.get('squares'),
                mask=batch.get('mask'),
                features=batch.get('features'),
                legal_mask=batch.get('legal_mask')
            )

            # Compute loss
            losses = self.loss_fn(outputs, batch)

            # Accumulate metrics
            total_loss += losses['loss'].item()
            total_policy_loss += losses.get('policy_loss', 0).item() if 'policy_loss' in losses else 0
            total_value_loss += losses.get('value_loss', 0).item() if 'value_loss' in losses else 0
            num_batches += 1

        metrics = {
            'val/loss': total_loss / num_batches,
            'val/policy_loss': total_policy_loss / num_batches,
            'val/value_loss': total_value_loss / num_batches
        }

        # Log to wandb
        if self.use_wandb:
            wandb.log({**metrics, 'epoch': self.current_epoch})

        return metrics

    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
            }, best_path)

        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)

        print(f"Loaded checkpoint from {checkpoint_path}")

    def train(
        self,
        num_epochs: int,
        save_every: int = 5,
        validate_every: int = 1
    ):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                config={
                    'model': self.model.__class__.__name__,
                    'num_params': self.model.count_parameters(),
                    'optimizer': self.optimizer.__class__.__name__,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'num_epochs': num_epochs
                }
            )

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch} - Train loss: {train_metrics['loss']:.4f}")

            # Validate
            if self.val_loader and (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                print(f"Epoch {epoch} - Val loss: {val_metrics['val/loss']:.4f}")

                # Save best model
                if val_metrics['val/loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val/loss']
                    self.save_checkpoint(is_best=True)
                    print("New best model!")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()

        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    print("Base trainer for chess models")
