import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
from config import CONFIG

# Set matplotlib backend for interactive pop-up windows in PyCharm Community
matplotlib.use('TkAgg')  # Use 'Qt5Agg' if TkAgg doesn't work

def visualize_results(csv_path, output_dir, class_names):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Print text-based classification report to CLI
    print("\nClassification Report:")
    for _, row in df.iterrows():
        case = row["case"]
        model = row["model"]
        print(f"\n{case} - {model}")
        report_data = {
            "Class": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": []
        }
        for class_name in class_names:
            report_data["Class"].append(class_name)
            report_data["Precision"].append(row.get(f"test_precision_{class_name}", 0.0))
            report_data["Recall"].append(row.get(f"test_recall_{class_name}", 0.0))
            report_data["F1 Score"].append(row.get(f"test_f1_{class_name}", 0.0))
        report_df = pd.DataFrame(report_data)
        print(report_df.to_string(index=False))
        print(f"Aggregated Metrics:")
        print(f"  Accuracy: {row['test_acc']:.3f}")
        print(f"  Precision (Macro): {row['test_precision']:.3f}")
        print(f"  Recall (Macro): {row['test_recall']:.3f}")
        print(f"  F1 Score (Macro): {row['test_f1']:.3f}")

    # Set seaborn style for professional look
    sns.set_style("whitegrid")
    sns.set_context("talk")  # Larger font for presentation

    # Aggregated metrics bar plots
    agg_metrics = ["test_acc", "test_precision", "test_recall", "test_f1"]
    agg_metric_names = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"]

    for metric, metric_name in zip(agg_metrics, agg_metric_names):
        plt.figure(figsize=(12, 6))
        sns.barplot(x="case", y=metric, hue="model", data=df)
        plt.title(f"Test {metric_name} by Case and Model")
        plt.xlabel("Augmentation Case")
        plt.ylabel(metric_name)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test_{metric}.png", dpi=300, bbox_inches="tight")
        plt.show()  # Display plot interactively; close window to proceed
        plt.close()

    # Classification report heatmap (per-class metrics)
    per_class_metrics = []
    for class_name in class_names:
        per_class_metrics.extend([
            f"test_precision_{class_name}",
            f"test_recall_{class_name}",
            f"test_f1_{class_name}"
        ])

    # Melt dataframe for per-class metrics
    per_class_df = df.melt(id_vars=["case", "model"], value_vars=per_class_metrics,
                           var_name="metric_class", value_name="value")
    per_class_df["metric"] = per_class_df["metric_class"].apply(lambda x: x.split("_")[1])  # precision, recall, f1
    per_class_df["class"] = per_class_df["metric_class"].apply(lambda x: "_".join(x.split("_")[2:]))  # class name

    # Create heatmap for each metric type
    for metric in ["precision", "recall", "f1"]:
        metric_df = per_class_df[per_class_df["metric"] == metric]
        pivot_table = metric_df.pivot_table(values="value", index=["case", "model"], columns="class")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"Test {metric.capitalize()} by Class, Case, and Model")
        plt.xlabel("Class")
        plt.ylabel("Case and Model")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test_{metric}_per_class_heatmap.png", dpi=300, bbox_inches="tight")
        plt.show()  # Display plot interactively; close window to proceed
        plt.close()


if __name__ == "__main__":
    csv_path = f"{CONFIG['output_path']}/results_log.csv"
    output_dir = f"{CONFIG['output_path']}/plots"
    class_names = ["noncrack", "sealedcrack", "sealedpatch", "thincrack"]  # Update if filtered
    visualize_results(csv_path, output_dir, class_names)