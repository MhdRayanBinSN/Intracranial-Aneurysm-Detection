"""
Main training entry point for Intracranial Aneurysm Detection.

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --debug --epochs 2
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data import AneurysmDataModule
from models import create_model, count_parameters
from training import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Aneurysm Detection Model')
    
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
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate',
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
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Debug mode
    if args.debug:
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 2
        print("Running in DEBUG mode")
    
    # Print configuration
    print("\n" + "=" * 60)
    print("INTRACRANIAL ANEURYSM DETECTION - TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Mixed precision: {config['training']['mixed_precision']}")
    print(f"  Model: {config['model']['name']}")
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
    print("\nLoading data...")
    data_module = AneurysmDataModule(config_path=str(config_path))
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = create_model(config)
    num_params = count_parameters(model)
    print(f"Model: {config['model']['name']}")
    print(f"Trainable parameters: {num_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
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
        trainer.save_checkpoint()
        print("Checkpoint saved.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
