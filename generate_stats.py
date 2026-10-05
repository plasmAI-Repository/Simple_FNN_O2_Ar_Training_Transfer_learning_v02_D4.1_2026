import argparse
import json
import os
import pandas as pd


def generate_statistics(csv_files, output_file="stats_min_max.json", sep=";"):
    """
    Computes min and max values for all numerical columns across the provided CSV files
    and saves the dictionary to a JSON file.
    """
    combined_df = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"Reading {csv_file}...")
            df = pd.read_csv(csv_file, sep=sep)
            combined_df.append(df)
        else:
            print(f"Warning: File {csv_file} does not exist.")

    if not combined_df:
        raise FileNotFoundError("No valid CSV files provided for statistics calculation.")

    df_all = pd.concat(combined_df, axis=0, ignore_index=True)
    stats = {}

    for column in df_all.columns:
        # Calculate min and max for numerical columns
        if pd.api.types.is_numeric_dtype(df_all[column]):
            min_val = float(df_all[column].min())
            max_val = float(df_all[column].max())
            stats[column] = {"min": min_val, "max": max_val}

    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\nSuccessfully generated statistics file: {output_file}")
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate min-max normalization statistics for dataset columns.")
    parser.add_argument("--csv_files", nargs="+", help="Paths to CSV dataset files (e.g. data/train_data_no_head_outer_corner_O2.csv)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing CSV files if csv_files is not explicitly passed")
    parser.add_argument("--output", type=str, default="stats_min_max.json", help="Output JSON path for stats")
    parser.add_argument("--sep", type=str, default=";", help="CSV delimiter (default ';')")

    args = parser.parse_args()

    csv_files = args.csv_files
    if not csv_files:
        # Default to finding all train CSV files in data_dir
        csv_files = [
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.endswith(".csv") and "train" in f
        ]
        if not csv_files:
            # Fallback to all CSVs
            csv_files = [
                os.path.join(args.data_dir, f)
                for f in os.listdir(args.data_dir)
                if f.endswith(".csv")
            ]

    print(f"Calculating min-max statistics for files: {csv_files}")
    generate_statistics(csv_files, output_file=args.output, sep=args.sep)


if __name__ == "__main__":
    main()
