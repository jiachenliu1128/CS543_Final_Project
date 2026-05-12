import pandas as pd
from pathlib import Path


def main():
    input_path = Path("data/cs543_data_collection_plan_updated.csv")
    output_path = Path("data/cs543_data_collection_plan_updated.csv")

    # Read CSV with UTF-8 encoding
    df = pd.read_csv(input_path, encoding="utf-8")

    # Compute physical areas
    df["Target Area (cm^2)"] = df["Target Length (cm)"] * df["Target Width (cm)"]
    df["Reference Area (cm^2)"] = df["Reference Length (cm)"] * df["Reference Width (cm)"]

    # Save with utf-8-sig so Excel can read special symbols correctly
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Updated file saved to: {output_path}")

if __name__ == "__main__":
    main()