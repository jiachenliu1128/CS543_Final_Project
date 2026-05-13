import pandas as pd
from pathlib import Path


def main():
    input_path = Path("data/cs543_data_collection_plan_updated.csv")
    output_path = Path("results/area_columns.csv")

    # Make sure the results folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the original metadata CSV
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Compute physical areas
    df["Target Area (cm^2)"] = df["Target Length (cm)"] * df["Target Width (cm)"]
    df["Reference Area (cm^2)"] = df["Reference Length (cm)"] * df["Reference Width (cm)"]

    # Keep only the columns needed for later evaluation
    output_df = df[
        [
            "Photo ID",
            "Image Filename",
            "Target Length (cm)",
            "Target Width (cm)",
            "Target Area (cm^2)",
            "Reference Length (cm)",
            "Reference Width (cm)",
            "Reference Area (cm^2)",
        ]
    ]

    # Save results separately
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Area results saved to: {output_path}")


if __name__ == "__main__":
    main()