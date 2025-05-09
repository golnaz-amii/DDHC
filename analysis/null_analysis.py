import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define datasets and directories
datasets = ["5m", "10m", "20m", "30m", "40m", "50m", "60m", "Variable"]
version = "Variable"
report_dir = "reports"
plot_dir = "plots/null_analysis"
os.makedirs(report_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

for version in ["Variable"]:
    dataset_path = rf'datasets/buffers/Labelled_StreamsV5_{version}.xls'
    df = pd.read_excel(dataset_path)

    # Display dataset information
    print(f"Dataset: {dataset_path}")
    print(df.info(), df.describe(), df.head(), df.columns, sep="\n\n")

    # print the number of unique values in buffer, and WatercourseRank, and if they match
    print(f"Unique values in 'Buffer': {df['Buffer'].nunique()}")
    print(f"Unique values in 'WatercourseRank': {df['WatercourseRank'].nunique()}")
    for buffer in df['Buffer'].unique():
        print(f"Buffer: {buffer}, WatercourseRank: {df[df['Buffer'] == buffer]['WatercourseRank'].unique()}")


    # Write null values report
    report_path = os.path.join(report_dir, f"null_report_{os.path.basename(dataset_path)}.txt")
    with open(report_path, "w") as report_file:
        report_file.write("Number of null values in each column:\n")
        for column, null_count in df.isnull().sum().items():
            report_file.write(f"{column:<30} {null_count:>10}\n")
        report_file.write("\n\nDataset Info:\n")
        df.info(buf=report_file)
        report_file.write("\n\nDataset Description:\n")
        report_file.write(df.describe().to_string())

    # Analyze and plot null values distribution
    null_rows = df[df.isnull().any(axis=1)]
    if not null_rows.empty:
        null_classes = null_rows["WatercourseRank"].value_counts()
        print("\nNull values distribution over classes:")
        print(null_classes)

        sns.countplot(x="WatercourseRank", data=null_rows, hue="WatercourseRank")
        plot_path = os.path.join(plot_dir, f"null_values_distribution_{os.path.basename(dataset_path)}.png")
        plt.savefig(plot_path)
        plt.close()
    else:
        print("\nNo null values found in the dataset.")
