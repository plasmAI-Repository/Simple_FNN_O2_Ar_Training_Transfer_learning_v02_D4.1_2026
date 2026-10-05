import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import torch
from torch.utils.data import DataLoader


from data_loader import MergedDatasetTest, unscale_min_max
from loss import WeightedMSE
from model import Model
from utils import setup_device, test_model, save_metrics_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PyTorch MLP regression model for Plasma Etch Rate prediction."
    )
    parser.add_argument("--gas", type=str, default="Ar", help="Chemistry / gas name (e.g. Ar, O2, CF4, SF6)")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "fine_tune", "freeze"],
        help="Training mode of model to test",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing test dataset CSV files")
    parser.add_argument("--stats_json", type=str, default="stats_min_max.json", help="Path to min-max statistics JSON file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save predictions and plots")
    parser.add_argument("--neurons_per_layer", type=int, default=10, help="Hidden layer size")
    parser.add_argument("--layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--no_plot_show", action="store_true", help="Do not call plt.show() interactively")
    return parser.parse_args()


def main():
    args = parse_args()

    device = setup_device()

    gas = args.gas
    mode = args.mode

    if args.output_dir:
        dir_path = args.output_dir
    else:
        dir_path = f"./results/{gas}/{mode}"
    os.makedirs(dir_path, exist_ok=True)

    csv_file_path_test = os.path.join(args.data_dir, f"test_data_no_head_outer_corner_{gas}.csv")
    if not os.path.exists(csv_file_path_test):
        all_csvs = [f for f in os.listdir(args.data_dir) if f.endswith('.csv') and gas.lower() in f.lower() and 'test' in f.lower()]
        if all_csvs:
            csv_file_path_test = os.path.join(args.data_dir, all_csvs[0])
        else:
            raise FileNotFoundError(f"Could not find test CSV for gas '{gas}' in {args.data_dir}")

    trained_pth = args.model_path if args.model_path else os.path.join(dir_path, "trained_model.pth")
    if not os.path.exists(trained_pth):
        raise FileNotFoundError(f"Model checkpoint not found at: {trained_pth}")

    print(f"Testing model checkpoint: {trained_pth}")
    print(f"Test data file: {csv_file_path_test}")

    dataset_test = MergedDatasetTest(csv_file_path_test, args.stats_json)
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    in_features = len(dataset_test.X_columns)
    out_features = len(dataset_test.y_columns)

    trained_model = Model(h1=args.neurons_per_layer, num_layers=args.layers, in_features=in_features, out_features=out_features)
    trained_model.load_state_dict(torch.load(trained_pth, map_location=device))

    criterion = WeightedMSE(reduction="mean", device=device)

    test_loss, all_predictions, all_targets, r2_scores_scaled, inference_time = test_model(
        model=trained_model, test_loader=test_loader, criterion=criterion, device=device
    )

    with open(args.stats_json, "r") as f:
        scaling_info = json.load(f)

    output_columns_names = dataset_test.y_columns

    unscaled_predictions = unscale_min_max(all_predictions, output_columns_names, scaling_info)
    predictions_df = pd.DataFrame(unscaled_predictions, columns=output_columns_names)
    predictions_df_csv = os.path.join(dir_path, "pred.csv")
    predictions_df.to_csv(predictions_df_csv, index=False, sep=";")
    print(f"Saved unscaled predictions to: {predictions_df_csv}")

    # Load unscaled ground truth targets for exact evaluation
    real_targets = pd.read_csv(csv_file_path_test, sep=";")[output_columns_names]

    r2_scores = []
    mae_scores = []
    mse_scores = []

    for col_name in output_columns_names:
        pred_col = predictions_df[col_name]
        target_col = real_targets[col_name]

        r2_scores.append(float(r2_score(target_col, pred_col)))
        mae_scores.append(float(mean_absolute_error(target_col, pred_col)))
        mse_scores.append(float(mean_squared_error(target_col, pred_col)))

    overall_unscaled_mse = float(mean_squared_error(real_targets.values, predictions_df.values))
    overall_unscaled_mae = float(mean_absolute_error(real_targets.values, predictions_df.values))
    overall_unscaled_r2 = float(r2_score(real_targets.values, predictions_df.values))

    # Save quantitative metrics to metrics.json
    metrics_summary = {
        "gas": gas,
        "mode": mode,
        "scaled_test_loss": float(test_loss),
        "overall_unscaled_r2": overall_unscaled_r2,
        "overall_unscaled_mae": overall_unscaled_mae,
        "overall_unscaled_mse": overall_unscaled_mse,
        "inference_time_seconds": float(inference_time),
        "point_r2_scores": dict(zip(output_columns_names, r2_scores)),
        "point_mae_scores": dict(zip(output_columns_names, mae_scores)),
        "point_mse_scores": dict(zip(output_columns_names, mse_scores)),
    }
    save_metrics_json(metrics_summary, os.path.join(dir_path, "metrics.json"))

    # 1. Bar Plot of R² Scores
    plt.figure(figsize=(10, 6))
    point_labels = [col.replace("_EtchRateO", "").replace("_EtchRate", "") for col in output_columns_names]
    plt.bar(list(range(1, len(r2_scores) + 1)), r2_scores, color="skyblue", edgecolor="black")
    plt.xlabel("Etching Rate Point")
    plt.ylabel(r"$R^2$ Score")
    plt.title(f"Spatial $R^2$ Scores per Point ({gas} - {mode})")
    plt.xticks(list(range(1, len(r2_scores) + 1)), point_labels, rotation=45 if len(output_columns_names) > 10 else 0)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    r2_plot_path = os.path.join(dir_path, "r2.png")
    plt.savefig(r2_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved R² plot to: {r2_plot_path}")
    plt.close()

    # 2. Residual Distribution Plot
    residuals = abs(real_targets - predictions_df) / real_targets * 100
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(data=residuals, fill=False)
    plt.title(f"Residual Percentage Distribution ({gas} - {mode})")
    plt.xlabel("|Physics-based - NN| / Physics-based (%)")
    plt.ylabel("Density / Frequency")
    plt.xlim(-1, 100)
    plt.tight_layout()
    residuals_plot_path = os.path.join(dir_path, "residuals.png")
    plt.savefig(residuals_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved residuals plot to: {residuals_plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
