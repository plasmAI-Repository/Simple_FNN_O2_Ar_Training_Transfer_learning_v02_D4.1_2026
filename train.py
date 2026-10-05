import argparse
import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

from data_loader import MergedDataset
from loss import WeightedMSE
from model import Model
from utils import setup_device, set_seed, train_regression_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PyTorch MLP regression model for Plasma Etch Rate prediction or Transfer Learning."
    )
    parser.add_argument("--gas", type=str, default="Ar", help="Target chemistry / gas name (e.g. Ar, O2, CF4, SF6)")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "fine_tune", "freeze"],
        help="Training mode: 'baseline' (from scratch), 'fine_tune' (full model fine-tuning), or 'freeze' (freeze initial layers)",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained model .pth file for transfer learning (required for fine_tune and freeze modes)",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing dataset CSV files")
    parser.add_argument("--stats_json", type=str, default="stats_min_max.json", help="Path to min-max statistics JSON file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save trained model and results")
    parser.add_argument("--epochs", type=int, default=700, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.000962055, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=2.055e-05, help="Optimizer weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=41, help="Random seed for reproducibility")
    parser.add_argument("--neurons_per_layer", type=int, default=10, help="Hidden layer size")
    parser.add_argument("--layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--freeze_layers", nargs="+", type=int, default=[], help="List of layer indices to freeze (e.g. 0 1)")
    parser.add_argument("--unfreeze", action="store_true", help="Whether to unfreeze layers at epoch 100")
    parser.add_argument("--no_plot_show", action="store_true", help="Do not call plt.show() interactively")
    return parser.parse_args()


def main():
    args = parse_args()

    device = setup_device()
    set_seed(args.seed)

    gas = args.gas
    mode = args.mode

    if args.output_dir:
        dir_path = args.output_dir
    else:
        dir_path = f"./results/{gas}/{mode}"
    os.makedirs(dir_path, exist_ok=True)

    csv_file_path_train = os.path.join(args.data_dir, f"train_data_no_head_outer_corner_{gas}.csv")

    if not os.path.exists(csv_file_path_train):
        # Fallback search if exact name doesn't match
        all_csvs = [f for f in os.listdir(args.data_dir) if f.endswith('.csv') and gas.lower() in f.lower() and 'train' in f.lower()]
        if all_csvs:
            csv_file_path_train = os.path.join(args.data_dir, all_csvs[0])
        else:
            raise FileNotFoundError(f"Could not find training CSV for gas '{gas}' in {args.data_dir}")

    print(f"Loading training data from: {csv_file_path_train}")
    dataset = MergedDataset(csv_file_path_train, statsfile=args.stats_json)

    in_features = len(dataset.X_columns)
    out_features = len(dataset.y_columns)
    print(f"Input Features ({in_features}): {dataset.X_columns}")
    print(f"Output Targets ({out_features}): {dataset.y_columns}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    freeze_layers = args.freeze_layers if mode == "freeze" else []
    model = Model(
        h1=args.neurons_per_layer,
        num_layers=args.layers,
        in_features=in_features,
        out_features=out_features,
        freeze_layers=freeze_layers,
    )

    lr = args.lr

    # Transfer Learning Pretrained Weight Loading
    if mode in ["fine_tune", "freeze"]:
        pretrained_path = args.pretrained_path
        if not pretrained_path:
            # Fallback path if not provided
            pretrained_path = "./results/O2/baseline/trained_model.pth"

        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
            if mode == "fine_tune":
                lr = lr / 20.0  # Reduce learning rate for fine-tuning source weights
                print(f"Fine-tuning mode: Learning rate adjusted to {lr}")
        else:
            print(f"Warning: Pretrained path '{pretrained_path}' not found! Training from scratch.")

    criterion = WeightedMSE(reduction="mean", device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1)

    start_time = time.time()
    trained_model, losses, val_losses, mean_r2 = train_regression_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        patience=args.patience,
        scheduler=scheduler,
        dir_path=dir_path,
        unfreeze_layers=args.unfreeze,
    )
    end_time = time.time()

    model_save_path = os.path.join(dir_path, "trained_model.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")
    print(f"Training time: {(end_time - start_time):.4f}s")
    print(f"Best Validation Mean R²: {mean_r2:.4f}")

    # Plot training loss progression
    plt.figure(figsize=(8, 5))
    plt.plot(list(range(len(losses))), losses, label="Training Loss")
    plt.plot(list(range(len(val_losses))), val_losses, label="Validation Loss")
    plt.ylabel("Total Loss")
    plt.xlabel("Epoch")
    plt.title(f"Loss Progression ({gas} - {mode})")
    plt.legend()
    plt.tight_layout()

    loss_fig_path = os.path.join(dir_path, "loss_progression.png")
    plt.savefig(loss_fig_path, dpi=300)
    print(f"Saved loss progression plot to {loss_fig_path}")

    if not args.no_plot_show:
        plt.close()


if __name__ == "__main__":
    main()
