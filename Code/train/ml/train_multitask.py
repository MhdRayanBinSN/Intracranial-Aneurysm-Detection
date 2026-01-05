"""
Main training entry point for Multi-Task Learning (Classification + Segmentation).

Usage:
    python train_multitask.py --config config/config.yaml
    python train_multitask.py --config config/config.yaml --debug --epochs 2
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data import MultiTaskDataModule
from models import MultiTaskUNet, count_parameters
from training import MultiTaskTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Task Aneurysm Detection + Segmentation Model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode with fewer samples',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training',
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Debug mode
    if args.debug:
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 1
        print("Running in DEBUG mode")
    
    # Print configuration
    print("\n" + "=" * 60)
    print("MULTI-TASK ANEURYSM DETECTION + SEGMENTATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Mixed precision: {config['training']['mixed_precision']}")
    print(f"  Classification weight: {config['training'].get('cls_weight', 1.0)}")
    print(f"  Segmentation weight: {config['training'].get('seg_weight', 1.0)}")
    print("=" * 60 + "\n")
    
    # Check CUDA
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize data module
    print("\nLoading segmentation data...")
    data_module = MultiTaskDataModule(config_path=str(config_path))
    data_module.setup_segmentation()
    
    train_loader = data_module.seg_train_dataloader()
    val_loader = data_module.seg_val_dataloader()
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\nInitializing multi-task model...")
    model = MultiTaskUNet(
        in_channels=config['model'].get('in_channels', 1),
        num_classes=config['model'].get('num_classes', 14),
        base_features=config['model'].get('base_features', 32),
        dropout=config['model'].get('dropout', 0.3),
    )
    num_params = count_parameters(model)
    print(f"Model: MultiTaskUNet")
    print(f"Trainable parameters: {num_params:,}")
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        best_metric = trainer.train()
        print(f"\n{'=' * 60}")
        print(f"Training completed successfully!")
        print(f"Best validation AUC: {best_metric:.4f}")
        print(f"{'=' * 60}")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted_multitask.pth')
        print("Checkpoint saved.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
