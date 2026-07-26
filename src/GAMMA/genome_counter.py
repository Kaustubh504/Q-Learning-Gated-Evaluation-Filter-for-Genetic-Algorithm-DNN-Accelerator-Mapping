import pandas as pd
import argparse


def count_unique_genomes(csv_path, out_xlsx):
    df = pd.read_csv(csv_path)

    df["fitness1"] = pd.to_numeric(df["fitness1"], errors="coerce")
    df["valid"] = df["valid"].astype(int)

    # ===== SAME LOGIC =====
    freq = df.groupby("genome_str").size().rename("count")
    best_fit = df.groupby("genome_str")["fitness1"].max().rename("best_fitness")
    pattern = df.groupby("genome_str")["genome_pattern"].first()

    summary = pd.concat([freq, best_fit, pattern], axis=1).reset_index()
    summary = summary.sort_values("best_fitness", ascending=False).reset_index(drop=True)

    # ===== PRINT =====
    print("\nTop 10 entries:\n")
    print(summary.head(10))

    # ===== SAVE TO EXCEL =====
    summary.to_excel(out_xlsx, index=False)
    print(f"\n✅ Saved Excel file → {out_xlsx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="genome_summary.xlsx")
    args = parser.parse_args()

    count_unique_genomes(args.csv, args.out)