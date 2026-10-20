
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Original dataset path
DATA_FILE = "Unemployment in India.csv"
# Output folder
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    print("=" * 55)
    print("       UNEMPLOYMENT DATA ANALYSIS")
    print("=" * 55)

    # Load original dataset
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}\n"
            "Please check the CSV file location."
        )

    df = pd.read_csv(DATA_FILE)

    # Clean column names and text fields
    df.columns = df.columns.str.strip()

    for col in ["Region", "Area", "Frequency"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    date_col = "Date"
    rate_col = "Estimated Unemployment Rate (%)"
    employed_col = "Estimated Employed"
    participation_col = "Estimated Labour Participation Rate (%)"

    required = ["Region", date_col, rate_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # Convert data types
    df[date_col] = pd.to_datetime(
        df[date_col], dayfirst=True, errors="coerce"
    )

    numeric_cols = [
        rate_col,
        employed_col,
        participation_col
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows that cannot be used for unemployment analysis
    df = df.dropna(subset=["Region", date_col, rate_col])
    df = df[df[rate_col] >= 0].copy()

    if df.empty:
        raise ValueError("No valid unemployment records found in the CSV.")

    df["Year-Month"] = df[date_col].dt.to_period("M").astype(str)

    # Save cleaned original records
    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_unemployment_data.csv")
    df.to_csv(cleaned_path, index=False)

    # 1. Monthly unemployment trend across available records
    monthly = (
        df.groupby("Year-Month", as_index=False)[rate_col]
        .mean()
        .sort_values("Year-Month")
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=monthly,
        x="Year-Month",
        y=rate_col,
        marker="o"
    )
    plt.title("Monthly Unemployment Rate Trend")
    plt.xlabel("Month")
    plt.ylabel("Average Unemployment Rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "unemployment_trend.png"),
        dpi=300
    )
    plt.close()

    # 2. Average unemployment rate by region
    regional = (
        df.groupby("Region")[rate_col]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        x=regional.values,
        y=regional.index
    )
    plt.title("Average Unemployment Rate by Region")
    plt.xlabel("Average Unemployment Rate (%)")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "unemployment_regional_comparison.png"),
        dpi=300
    )
    plt.close()

    # 3. Rural vs Urban comparison, if Area exists in the dataset
    area_chart_created = False

    if "Area" in df.columns:
        area_data = df.dropna(subset=["Area"])

        if not area_data.empty:
            area_avg = (
                area_data.groupby("Area")[rate_col]
                .mean()
                .sort_values(ascending=False)
            )

            plt.figure(figsize=(8, 5))
            sns.barplot(
                x=area_avg.index,
                y=area_avg.values
            )
            plt.title("Average Unemployment Rate by Area")
            plt.xlabel("Area")
            plt.ylabel("Average Unemployment Rate (%)")
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, "unemployment_area_comparison.png"),
                dpi=300
            )
            plt.close()
            area_chart_created = True

    # Calculate key findings
    highest_month = monthly.loc[monthly[rate_col].idxmax()]
    lowest_month = monthly.loc[monthly[rate_col].idxmin()]
    highest_region = regional.idxmax()
    lowest_region = regional.idxmin()

    # Save a readable report
    report_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("UNEMPLOYMENT DATA ANALYSIS SUMMARY\n")
        report.write("=" * 40 + "\n")
        report.write(f"Source file: {DATA_FILE}\n")
        report.write(f"Valid records analyzed: {len(df)}\n")
        report.write(f"Regions included: {df['Region'].nunique()}\n")
        report.write(
            f"Date range: {df[date_col].min().date()} "
            f"to {df[date_col].max().date()}\n\n"
        )
        report.write(
            f"Overall average unemployment rate: "
            f"{df[rate_col].mean():.2f}%\n"
        )
        report.write(
            f"Highest monthly average: {highest_month[rate_col]:.2f}% "
            f"({highest_month['Year-Month']})\n"
        )
        report.write(
            f"Lowest monthly average: {lowest_month[rate_col]:.2f}% "
            f"({lowest_month['Year-Month']})\n"
        )
        report.write(
            f"Highest-average unemployment region: "
            f"{highest_region} ({regional[highest_region]:.2f}%)\n"
        )
        report.write(
            f"Lowest-average unemployment region: "
            f"{lowest_region} ({regional[lowest_region]:.2f}%)\n"
        )

        if "Area" in df.columns:
            report.write("\nAverage unemployment rate by area:\n")
            for area, value in df.groupby("Area")[rate_col].mean().items():
                report.write(f"  {area}: {value:.2f}%\n")

    # Display results in terminal
    print("\n===== DATASET INFORMATION =====")
    print(f"Valid records: {len(df)}")
    print(f"Regions: {df['Region'].nunique()}")
    print(f"Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")

    print("\n===== REGIONAL UNEMPLOYMENT RATES =====")
    print(regional.round(2).to_string())

    print("\n===== MONTHLY TREND =====")
    print(monthly.to_string(index=False))

    print("\n===== PROJECT OUTPUTS =====")
    print(f"Cleaned data: {cleaned_path}")
    print(f"Trend chart: {os.path.join(OUTPUT_DIR, 'unemployment_trend.png')}")
    print(
        "Regional chart: "
        f"{os.path.join(OUTPUT_DIR, 'unemployment_regional_comparison.png')}"
    )

    if area_chart_created:
        print(
            "Area chart: "
            f"{os.path.join(OUTPUT_DIR, 'unemployment_area_comparison.png')}"
        )

    print(f"Summary report: {report_path}")
    print("\nSUCCESS: Analysis completed using the original CSV dataset!")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        raise
