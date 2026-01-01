"""
Main training script for chess models.

Usage:
    python -m training.train --config configs/direction_a.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path

from models.direction_a import ChessTransformer
from models.direction_b import DenseNNUE, HybridEngine
from models.direction_c import ChessMoE
from models.direction_d import ParallelVerificationChess

from data.datasets import create_dataloaders
from training.trainers.base_trainer import BaseTrainer
from training.losses import ChessLoss, MoELoss, ParallelVerificationLoss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> nn.Module:
    """Create model based on config."""
    model_type = config['model']['type']
    model_params = config['model']['params']

    if model_type == 'ChessTransformer':
        return ChessTransformer(**model_params)
    elif model_type == 'DenseNNUE':
        return DenseNNUE(**model_params)
    elif model_type == 'HybridEngine':
        # Create eval and policy nets separately
        eval_params = model_params.get('eval_net', {})
        policy_params = model_params.get('policy_net', {})

        from models.direction_b.nnue_models import DenseNNUE, PolicyNetwork
        eval_net = DenseNNUE(**eval_params) if eval_params else None
        policy_net = PolicyNetwork(**policy_params) if policy_params else None

        return HybridEngine(eval_net=eval_net, policy_net=policy_net)
    elif model_type == 'ChessMoE':
        return ChessMoE(**model_params)
    elif model_type == 'ParallelVerificationChess':
        return ParallelVerificationChess(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_loss_fn(config: dict) -> nn.Module:
    """Create loss function based on config."""
    loss_config = config['training']['loss']

    loss_type = loss_config.get('type', 'ChessLoss')

    if loss_type == 'ChessLoss':
        return ChessLoss(
            policy_weight=loss_config.get('policy_weight', 0.5),
            value_weight=loss_config.get('value_weight', 0.5),
            value_loss_type=loss_config.get('value_loss_type', 'mse')
        )
    elif loss_type == 'MoELoss':
        return MoELoss(
            policy_weight=loss_config.get('policy_weight', 0.5),
            value_weight=loss_config.get('value_weight', 0.5),
            balance_weight=loss_config.get('balance_weight', 0.01),
            value_loss_type=loss_config.get('value_loss_type', 'mse')
        )
    elif loss_type == 'ParallelVerificationLoss':
        return ParallelVerificationLoss(
            policy_weight=loss_config.get('policy_weight', 0.5),
            value_weight=loss_config.get('value_weight', 0.5),
            process_reward_weight=loss_config.get('process_reward_weight', 0.3),
            value_loss_type=loss_config.get('value_loss_type', 'mse')
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    optimizer_type = config['training'].get('optimizer', 'AdamW')
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.01)

    if optimizer_type == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    num_training_steps: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    scheduler_type = config['training'].get('scheduler', 'cosine')
    warmup_steps = config['training'].get('warmup_steps', 0)

    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=1e-6
        )
    elif scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def main(args):
    """Main training function."""
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(yaml.dump(config, default_flow_style=False))

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        batch_size=config['data']['batch_size'],
        encoding_type=config['data']['encoding_type'],
        num_workers=config['data']['num_workers'],
        max_train_samples=config['data'].get('max_train_samples'),
        max_val_samples=config['data'].get('max_val_samples')
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model.print_model_stats()

    # Create loss function
    loss_fn = create_loss_fn(config)
    print(f"\nUsing loss: {loss_fn.__class__.__name__}")

    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"Using optimizer: {optimizer.__class__.__name__}")

    # Create scheduler
    num_epochs = config['training']['num_epochs']
    num_training_steps = num_epochs * len(train_loader)
    scheduler = create_scheduler(optimizer, config, num_training_steps)

    if scheduler:
        print(f"Using scheduler: {scheduler.__class__.__name__}")

    # Create trainer
    checkpoint_dir = config['training']['checkpoint_dir']
    use_wandb = config['wandb']['enabled']
    wandb_project = config['wandb']['project']

    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project
    )

    # Load checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\nStarting training for {num_epochs} epochs...")
    trainer.train(
        num_epochs=num_epochs,
        save_every=config['training']['save_every'],
        validate_every=config['training']['validate_every']
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chess models")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()
    main(args)
