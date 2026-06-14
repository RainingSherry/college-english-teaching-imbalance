from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

from latex_cv_experiment import (
    CLASS_LABELS_ZH,
    FEATURE_COLUMNS,
    SEED,
    configure_chinese_font,
    gan_resample,
    set_seed,
)


LABEL_ORDER = ["高质量", "中质量", "低质量"]
DISTRIBUTION_ORDER = ["中质量", "低质量", "高质量"]
STYLE_MAP = {
    "高质量": {"color": "red", "marker": "^"},
    "中质量": {"color": "green", "marker": "s"},
    "低质量": {"color": "blue", "marker": "o"},
}


def center_distance(points: np.ndarray, labels: np.ndarray, label_a: str, label_b: str) -> float:
    center_a = points[labels == label_a].mean(axis=0)
    center_b = points[labels == label_b].mean(axis=0)
    return float(np.linalg.norm(center_a - center_b))


def update_report(report_path: Path, row: dict[str, object]) -> None:
    if report_path.exists():
        report_df = pd.read_csv(report_path, encoding="utf-8-sig")
        report_df = report_df[~report_df["策略"].isin([row["策略"], "GAN插补"])]
    else:
        report_df = pd.DataFrame()
    report_df = pd.concat([report_df, pd.DataFrame([row])], ignore_index=True)
    order = {"No Sampling": 0, "SMOTE": 1, "BorderlineSMOTE": 2, "SMOTEENN": 3, "GAN": 4}
    report_df["_order"] = report_df["策略"].map(order).fillna(99)
    report_df = report_df.sort_values("_order").drop(columns="_order")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")


def plot_gan_tsne() -> None:
    set_seed(SEED)
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent
    result_dir = project_dir / "03_结果"
    data_path = project_dir / "02_数据" / "college_english_teaching_evaluation.csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    configure_chinese_font(project_dir)

    df = pd.read_csv(data_path)
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["teaching_quality_label"])
    high_label = int(label_encoder.transform(["High"])[0])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_gan, y_gan = gan_resample(x_scaled, y, high_label=high_label, random_state=SEED)
    label_names = label_encoder.inverse_transform(y_gan)
    labels_zh = np.array([CLASS_LABELS_ZH[label] for label in label_names])

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    embedding = tsne.fit_transform(x_gan)

    counts = Counter(labels_zh)
    total = len(labels_zh)
    fig, ax = plt.subplots(figsize=(12, 10))
    for label in LABEL_ORDER:
        mask = labels_zh == label
        style = STYLE_MAP[label]
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=style["color"],
            marker=style["marker"],
            label=f"{label}（{counts[label]}）",
            alpha=0.7,
            edgecolors="white",
            linewidths=0.45,
            s=80,
        )

    distribution_lines = ["类别分布:"]
    for label in DISTRIBUTION_ORDER:
        distribution_lines.append(f"{label}: {counts[label]:>3} ({counts[label] / total * 100:4.1f}%)")
    ax.text(
        0.02,
        0.98,
        "\n".join(distribution_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "wheat", "edgecolor": "black", "alpha": 0.75},
    )

    ax.set_title(f"GAN - t-SNE特征空间可视化\n(总样本数： {total})", fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE维度1", fontsize=13)
    ax.set_ylabel("t-SNE维度2", fontsize=13)
    ax.grid(alpha=0.3)
    legend = ax.legend(title="质量标签（样本数）", loc="upper right", fontsize=12, title_fontsize=13)
    legend.get_title().set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(result_dir / "tsne_gan.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_row = {
        "策略": "GAN",
        "样本数": total,
        "高质量": counts["高质量"],
        "中质量": counts["中质量"],
        "低质量": counts["低质量"],
        "高vs中距离": round(center_distance(embedding, labels_zh, "高质量", "中质量"), 3),
        "高vs低距离": round(center_distance(embedding, labels_zh, "高质量", "低质量"), 3),
        "中vs低距离": round(center_distance(embedding, labels_zh, "中质量", "低质量"), 3),
    }
    update_report(result_dir / "sampling_tsne_comparison_report.csv", report_row)
    print(f"已生成: {result_dir / 'tsne_gan.png'}")
    print(f"GAN插补类别分布: {dict(counts)}")


if __name__ == "__main__":
    plot_gan_tsne()
