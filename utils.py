import torch
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import time
import json
import os


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}.')
    return device


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_regression_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience,
                           scheduler, dir_path, verbose=False, unfreeze_layers=False, layers_to_unfreeze=[2]):
    model.to(device)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    early_counter = 0
    best_model_state = None
    best_df_val = None

    for epoch in range(num_epochs):

        if epoch == 100 and unfreeze_layers:
            for layer_idx in layers_to_unfreeze:
                if 0 <= layer_idx < len(model.layers):
                    for param in model.layers[layer_idx].parameters():
                        param.requires_grad = True

            # Adjust learning rate upon unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

            print(f"Layers {layers_to_unfreeze} unfrozen at epoch {epoch} and learning rate reduced by half.")

        model.train()
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            train_loss += loss.item() * inputs.size(0)

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation step
        val_loss = 0
        r2_values, mse_values = [], []
        all_predictions, all_targets = [], []

        model.eval()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_outputs = model(inputs)
                loss_val = criterion(val_outputs, targets)
                val_loss += loss_val.item() * inputs.size(0)

                all_predictions.append(val_outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        for col in range(all_targets.shape[1]):
            pred_col = all_predictions[:, col]
            target_col = all_targets[:, col]
            r2_values.append(r2_score(target_col, pred_col))
            mse_values.append(mean_squared_error(target_col, pred_col))

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        mean_r2 = float(np.mean(r2_values))
        mean_mse = float(np.mean(mse_values))

        if scheduler:
            scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val Mean R²: {mean_r2:.4f} | "
                  f"Val Mean MSE: {mean_mse:.6f}")

            if verbose:
                for col in range(all_targets.shape[1]):
                    print(f"  Col {col + 1} -> R²: {r2_values[col]:.4f}, MSE: {mse_values[col]:.6f}")

        # Early stopping & best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            best_model_state = model.state_dict()

            num_targets = all_targets.shape[1]
            pred_columns = [f'Pred_{i + 1}' for i in range(num_targets)]
            target_columns = [f'Target_{i + 1}' for i in range(num_targets)]
            best_df_val = pd.DataFrame(np.hstack((all_predictions, all_targets)), columns=pred_columns + target_columns)
        else:
            early_counter += 1
            if early_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    if best_df_val is not None and dir_path:
        os.makedirs(dir_path, exist_ok=True)
        best_df_val.to_csv(os.path.join(dir_path, 'val_set_results.csv'), index=False, sep=';')

    return model, train_losses, val_losses, mean_r2


def test_model(model, test_loader, criterion, device):
    model.to(device)
    model.eval()

    test_loss = 0
    r2_values, mse_values = [], []
    all_predictions, all_targets = [], []

    with torch.no_grad():
        start_time = time.time()
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        end_time = time.time()

    elapsed_time = end_time - start_time
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    for col in range(all_targets.shape[1]):
        pred_col = all_predictions[:, col]
        target_col = all_targets[:, col]
        r2_values.append(r2_score(target_col, pred_col))
        mse_values.append(mean_squared_error(target_col, pred_col))

    overall_mse = float(mean_squared_error(all_targets, all_predictions))
    overall_mae = float(mean_absolute_error(all_targets, all_predictions))
    overall_r2 = float(r2_score(all_targets, all_predictions))

    print(f"\nOverall Performance Metrics:")
    print(f"  Mean MSE: {overall_mse:.7f}")
    print(f"  Mean MAE: {overall_mae:.7f}")
    print(f"  Mean R²: {overall_r2:.5f}")
    print(f"  Inference Time: {elapsed_time:.4f}s")

    test_loss /= len(test_loader.dataset)

    return test_loss, all_predictions, all_targets, r2_values, elapsed_time


def save_metrics_json(metrics_dict, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved evaluation metrics to {file_path}")


def unscale(data, column_names, scaling_info):
    unscaled_data = np.copy(data)
    num_columns = data.shape[1]
    for i, column in enumerate(column_names):
        if i >= num_columns:
            print(f"Warning: Index {i} is out of bounds for data with {num_columns} columns.")
            break
        mean = scaling_info[column]['mean']
        std = scaling_info[column]['std']
        unscaled_data[:, i] = (data[:, i] * std) + mean
    return unscaled_data

