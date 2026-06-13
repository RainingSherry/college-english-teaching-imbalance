import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn


SEED = 42
FEATURE_COLUMNS = [
    "attendance_rate",
    "assignment_completion_rate",
    "avg_assignment_score",
    "exam_score",
    "quiz_avg_score",
    "participation_frequency",
    "lms_logins",
    "avg_session_duration",
    "video_completion_rate",
    "self_assessment_score",
    "peer_feedback_score",
    "teacher_feedback_score",
    "engagement_index",
]
SAMPLING_METHODS = ["NoSampling", "SMOTE", "SMOTEENN", "BorderlineSMOTE", "GAN"]
MODEL_FACTORIES = {
    "Decision Tree": lambda seed: DecisionTreeClassifier(random_state=seed, class_weight=None),
    "Random Forest": lambda seed: RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1, class_weight=None
    ),
    "SVM": lambda seed: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
    "KNN": lambda seed: KNeighborsClassifier(n_neighbors=13),
    "Logistic Regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
}
METHOD_LABELS_ZH = {
    "NoSampling": "不采样",
    "SMOTE": "SMOTE",
    "SMOTEENN": "SMOTE-ENN",
    "BorderlineSMOTE": "边界SMOTE",
    "GAN": "GAN插补",
}
METHOD_LABELS_EN = {
    "NoSampling": "NoSampling",
    "SMOTE": "SMOTE",
    "SMOTEENN": "SMOTEENN",
    "BorderlineSMOTE": "Borderline",
    "GAN": "GAN",
}
MODEL_LABELS_ZH = {
    "Decision Tree": "决策树",
    "Random Forest": "随机森林",
    "SVM": "支持向量机",
    "KNN": "K近邻",
    "Logistic Regression": "逻辑回归",
}
MODEL_LABELS_EN = {
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "SVM": "SVM",
    "KNN": "KNN",
    "Logistic Regression": "LR",
}
CLASS_LABELS_ZH = {
    "High": "高质量",
    "Medium": "中质量",
    "Low": "低质量",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def configure_chinese_font(project_dir: Path) -> None:
    candidates = [
        *sorted((project_dir / "04_latex" / "fonts").glob("*.otf")),
        *sorted(Path("/tmp/tectonic-cache").glob("**/FandolSong-Regular.otf")),
        *sorted(Path("/tmp/tectonic-cache").glob("**/FandolHei-Regular.otf")),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def geometric_mean(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    recalls = []
    for label in labels:
        mask = y_true == label
        if np.any(mask):
            recalls.append(accuracy_score(y_true[mask], y_pred[mask]))
    return float(np.prod(recalls) ** (1 / len(recalls))) if recalls else 0.0


def single_label_f1(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    return f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    high_label: int,
    medium_label: int,
    low_label: int,
) -> dict:
    return {
        "High_F1": single_label_f1(y_true, y_pred, high_label),
        "Medium_F1": single_label_f1(y_true, y_pred, medium_label),
        "Low_F1": single_label_f1(y_true, y_pred, low_label),
        "High_Precision": precision_score(
            y_true, y_pred, labels=[high_label], average="macro", zero_division=0
        ),
        "High_Recall": recall_score(
            y_true, y_pred, labels=[high_label], average="macro", zero_division=0
        ),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Balanced_Accuracy": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "G_mean": geometric_mean(y_true, y_pred, labels),
    }


def score_matrix(model, x_test: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw_scores = model.predict_proba(x_test)
    else:
        raw_scores = model.decision_function(x_test)
    raw_scores = np.asarray(raw_scores)
    if raw_scores.ndim == 1:
        raw_scores = raw_scores.reshape(-1, 1)

    scores = np.zeros((len(x_test), len(labels)), dtype=float)
    for source_idx, class_label in enumerate(model.classes_):
        target_idx = int(np.where(labels == class_label)[0][0])
        if source_idx < raw_scores.shape[1]:
            scores[:, target_idx] = raw_scores[:, source_idx]
    return scores


def smote_generate(
    class_x: np.ndarray,
    n_generate: int,
    rng: np.random.Generator,
    base_pool: np.ndarray | None = None,
) -> np.ndarray:
    if n_generate <= 0:
        return np.empty((0, class_x.shape[1]), dtype=class_x.dtype)
    if len(class_x) < 2:
        return class_x[rng.integers(0, len(class_x), size=n_generate)]

    pool = base_pool if base_pool is not None and len(base_pool) >= 1 else class_x
    n_neighbors = min(6, len(class_x))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    nn_model.fit(class_x)
    neighbor_indices = nn_model.kneighbors(pool, return_distance=False)
    base_indices = rng.integers(0, len(pool), size=n_generate)
    synthetic = []
    for pool_idx in base_indices:
        candidates = neighbor_indices[pool_idx][1:]
        if len(candidates) == 0:
            neighbor = pool[pool_idx]
        else:
            neighbor = class_x[int(rng.choice(candidates))]
        gap = rng.random()
        synthetic.append(pool[pool_idx] + gap * (neighbor - pool[pool_idx]))
    return np.asarray(synthetic, dtype=class_x.dtype)


def smote_resample(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
    target_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    counts = Counter(y)
    target = target_count or max(counts.values())
    x_parts = [x]
    y_parts = [y]
    for label, count in sorted(counts.items()):
        need = target - count
        if need <= 0:
            continue
        class_x = x[y == label]
        synthetic = smote_generate(class_x, need, rng)
        x_parts.append(synthetic)
        y_parts.append(np.full(need, label, dtype=y.dtype))
    return np.vstack(x_parts), np.concatenate(y_parts)


def borderline_smote_resample(x: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    counts = Counter(y)
    target = max(counts.values())
    x_parts = [x]
    y_parts = [y]
    global_nn = NearestNeighbors(n_neighbors=min(11, len(x)))
    global_nn.fit(x)
    global_neighbors = global_nn.kneighbors(x, return_distance=False)

    for label, count in sorted(counts.items()):
        need = target - count
        if need <= 0:
            continue
        class_indices = np.flatnonzero(y == label)
        danger_indices = []
        for idx in class_indices:
            neighbors = global_neighbors[idx][1:]
            majority_count = int(np.sum(y[neighbors] != label))
            if 0 < majority_count < len(neighbors):
                danger_indices.append(idx)
        base_pool = x[danger_indices] if danger_indices else x[class_indices]
        synthetic = smote_generate(x[class_indices], need, rng, base_pool=base_pool)
        x_parts.append(synthetic)
        y_parts.append(np.full(need, label, dtype=y.dtype))
    return np.vstack(x_parts), np.concatenate(y_parts)


def enn_clean(x: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= n_neighbors + 1:
        return x, y
    nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_model.fit(x)
    neighbors = nn_model.kneighbors(x, return_distance=False)[:, 1:]
    keep = np.ones(len(x), dtype=bool)
    for i, neighbor_idx in enumerate(neighbors):
        labels, counts = np.unique(y[neighbor_idx], return_counts=True)
        majority_label = labels[np.argmax(counts)]
        if majority_label != y[i]:
            keep[i] = False
    return x[keep], y[keep]


def smoteenn_resample(x: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    x_smote, y_smote = smote_resample(x, y, random_state=random_state)
    return enn_clean(x_smote, y_smote, n_neighbors=3)


class Generator(nn.Module):
    def __init__(self, noise_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def gradient_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    scores = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_minority_wgan(
    minority_x: np.ndarray,
    n_generate: int,
    random_state: int,
    epochs: int = 350,
    noise_dim: int = 16,
) -> np.ndarray:
    if n_generate <= 0:
        return np.empty((0, minority_x.shape[1]), dtype=minority_x.dtype)
    set_seed(random_state)
    device = torch.device("cpu")
    data = torch.tensor(minority_x, dtype=torch.float32, device=device)
    batch_size = min(32, len(data))
    generator = Generator(noise_dim, minority_x.shape[1]).to(device)
    discriminator = Discriminator(minority_x.shape[1]).to(device)
    g_opt = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.9))
    rng = np.random.default_rng(random_state)

    for _ in range(epochs):
        for _ in range(3):
            indices = rng.integers(0, len(data), size=batch_size)
            real = data[indices]
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake = generator(noise).detach()
            d_loss = discriminator(fake).mean() - discriminator(real).mean()
            d_loss = d_loss + 10.0 * gradient_penalty(discriminator, real, fake)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

        noise = torch.randn(batch_size, noise_dim, device=device)
        g_loss = -discriminator(generator(noise)).mean()
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    with torch.no_grad():
        synthetic = generator(torch.randn(n_generate, noise_dim, device=device)).cpu().numpy()
    lower = minority_x.min(axis=0) - 0.25 * minority_x.std(axis=0)
    upper = minority_x.max(axis=0) + 0.25 * minority_x.std(axis=0)
    return np.clip(synthetic, lower, upper).astype(minority_x.dtype)


def gan_resample(
    x: np.ndarray,
    y: np.ndarray,
    high_label: int,
    random_state: int,
    high_target_count: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(y)
    need = high_target_count - counts[high_label]
    if need <= 0:
        return x, y
    synthetic_high = train_minority_wgan(x[y == high_label], need, random_state=random_state)
    return np.vstack([x, synthetic_high]), np.concatenate([y, np.full(need, high_label, dtype=y.dtype)])


def resample_train(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    high_label: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "SMOTE":
        return smote_resample(x, y, random_state=random_state)
    if method == "BorderlineSMOTE":
        return borderline_smote_resample(x, y, random_state=random_state)
    if method == "SMOTEENN":
        return smoteenn_resample(x, y, random_state=random_state)
    if method == "GAN":
        return gan_resample(x, y, high_label=high_label, random_state=random_state)
    return x, y


def style_best_second(values: list[float], labels: list[str]) -> dict[str, str]:
    unique_desc = sorted(set(round(v, 12) for v in values), reverse=True)
    best = unique_desc[0]
    second = unique_desc[1] if len(unique_desc) > 1 else None
    styled = {}
    for label, value in zip(labels, values):
        text = f"{value:.3f}"
        rounded = round(value, 12)
        if rounded == best:
            text = f"\\textbf{{{text}}}"
        elif second is not None and rounded == second:
            text = f"\\underline{{{text}}}"
        styled[label] = text
    return styled


def style_best_second_mean_var(
    mean_values: list[float],
    var_values: list[float],
    labels: list[str],
) -> dict[str, str]:
    unique_desc = sorted(set(round(v, 12) for v in mean_values), reverse=True)
    best = unique_desc[0]
    second = unique_desc[1] if len(unique_desc) > 1 else None
    styled = {}
    for label, mean_value, var_value in zip(labels, mean_values, var_values):
        text = f"{mean_value:.3f}$\\pm${var_value:.3f}"
        rounded = round(mean_value, 12)
        if rounded == best:
            text = f"\\textbf{{{text}}}"
        elif second is not None and rounded == second:
            text = f"\\underline{{{text}}}"
        styled[label] = text
    return styled


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def build_latex_table(summary_df: pd.DataFrame, tex_path: Path) -> None:
    ordered = summary_df.sort_values(["High_F1_mean", "Macro_F1_mean"], ascending=False).reset_index(drop=True)
    combo_labels = [f"{row.Model_ZH}+{row.Method_ZH}" for row in ordered.itertuples()]
    metric_cols = ["High_F1_mean", "High_Recall_mean", "Macro_F1_mean", "G_mean_mean"]
    styled_cols = {
        col: style_best_second_mean_var(
            ordered[col].tolist(),
            ordered[f"{col[:-5]}_var"].tolist(),
            combo_labels,
        )
        for col in metric_cols
    }

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{不同模型与采样/插补方式下的5折交叉验证结果（均值$\\pm$方差）}",
        "\\label{tab:cv-results}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.6pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "模型 & 采样/插补方式 & 高质量F1 & 高质量召回率 & 宏平均F1 & G-均值 \\\\",
        "\\midrule",
    ]
    for row in ordered.itertuples():
        combo = f"{row.Model_ZH}+{row.Method_ZH}"
        model_text = latex_escape(row.Model_ZH)
        method_text = latex_escape(row.Method_ZH)
        if styled_cols["High_F1_mean"][combo].startswith("\\textbf"):
            model_text = f"\\textbf{{{model_text}}}"
            method_text = f"\\textbf{{{method_text}}}"
        elif styled_cols["High_F1_mean"][combo].startswith("\\underline"):
            model_text = f"\\underline{{{model_text}}}"
            method_text = f"\\underline{{{method_text}}}"
        lines.append(
            f"{model_text} & {method_text} & "
            f"{styled_cols['High_F1_mean'][combo]} & "
            f"{styled_cols['High_Recall_mean'][combo]} & "
            f"{styled_cols['Macro_F1_mean'][combo]} & "
            f"{styled_cols['G_mean_mean'][combo]} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\vspace{2pt}",
            "\\parbox{\\textwidth}{\\footnotesize 注：表中数值为5折交叉验证的均值$\\pm$方差，方差按5折结果计算。各指标列中最优均值加粗，第二优均值加下划线；模型与采样/插补方式名称的加粗或下划线依据高质量F1均值排序。}",
            "\\end{table}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(summary_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    method_order = SAMPLING_METHODS
    marker_cycle = ["o", "s", "^", "D", "P"]
    for marker, (model, group) in zip(marker_cycle, summary_df.groupby("Model")):
        group = group.set_index("Method").loc[method_order]
        ax.plot(
            [METHOD_LABELS_EN[m] for m in method_order],
            group[metric],
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=MODEL_LABELS_EN[model],
        )
    ax.set_xlabel("Sampling / interpolation strategy")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_boxplot(raw_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    method_order = SAMPLING_METHODS
    model_order = list(MODEL_FACTORIES.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    base_positions = np.arange(len(method_order), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(model_order))
    width = 0.105

    for model_index, model_name in enumerate(model_order):
        values = []
        positions = []
        for method_index, method in enumerate(method_order):
            subset = raw_df[(raw_df["Model"] == model_name) & (raw_df["Method"] == method)]
            values.append(subset[metric].to_numpy(dtype=float))
            positions.append(base_positions[method_index] + offsets[model_index])

        box = ax.boxplot(
            values,
            positions=positions,
            widths=width,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": colors[model_index % len(colors)],
                "markersize": 3.5,
            },
            medianprops={"color": "black", "linewidth": 0.8},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"markersize": 2.5},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[model_index % len(colors)])
            patch.set_alpha(0.55)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i % len(colors)],
            marker="s",
            linestyle="",
            markersize=7,
            label=MODEL_LABELS_ZH[model_name],
        )
        for i, model_name in enumerate(model_order)
    ]
    ax.set_xticks(base_positions)
    ax.set_xticklabels([METHOD_LABELS_ZH[m] for m in method_order])
    ax.set_xlabel("采样/插补方式")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=handles, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_roc_panel(
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    auc_path: Path,
    top_n: int = 6,
) -> None:
    top_combos = summary_df.sort_values(["High_F1_mean", "Macro_F1_mean"], ascending=False).head(top_n)
    combo_keys = [(row.Model, row.Method) for row in top_combos.itertuples()]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    auc_rows = []

    fig, axes = plt.subplots(1, len(class_names), figsize=(13.0, 4.0), sharex=True, sharey=True)
    if len(class_names) == 1:
        axes = [axes]

    for axis, class_name in zip(axes, class_names):
        score_column = f"Score_{class_name}"
        for combo_idx, (model_name, method_name) in enumerate(combo_keys):
            subset = score_df[(score_df["Model"] == model_name) & (score_df["Method"] == method_name)]
            if subset.empty:
                continue
            y_combo_true = (subset["True_Label_Name"] == class_name).astype(int).to_numpy()
            y_score = subset[score_column].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y_combo_true, y_score)
            roc_auc = auc(fpr, tpr)
            label = f"{MODEL_LABELS_ZH[model_name]}+{METHOD_LABELS_ZH[method_name]} AUC={roc_auc:.3f}"
            axis.plot(fpr, tpr, linewidth=1.45, color=colors[combo_idx % len(colors)], label=label)
            auc_rows.append(
                {
                    "类别": CLASS_LABELS_ZH[class_name],
                    "模型": MODEL_LABELS_ZH[model_name],
                    "采样/插补方式": METHOD_LABELS_ZH[method_name],
                    "AUC": round(float(roc_auc), 3),
                }
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=0.9)
        axis.set_title(f"{CLASS_LABELS_ZH[class_name]}类别")
        axis.set_xlabel("假阳性率")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6.2, loc="lower right", framealpha=0.92)

    axes[0].set_ylabel("真阳性率")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame(auc_rows).to_csv(auc_path, index=False, encoding="utf-8-sig")


def main() -> None:
    set_seed(SEED)
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent
    data_path = project_dir / "02_数据" / "college_english_teaching_evaluation.csv"
    result_dir = project_dir / "03_结果"
    latex_dir = project_dir / "04_latex"
    figure_dir = latex_dir / "figures"
    table_dir = latex_dir / "tables"
    for directory in [result_dir, latex_dir, figure_dir, table_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    configure_chinese_font(project_dir)

    df = pd.read_csv(data_path)
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["teaching_quality_label"])
    labels = encoder.transform(encoder.classes_)
    high_label = int(encoder.transform(["High"])[0])
    medium_label = int(encoder.transform(["Medium"])[0])
    low_label = int(encoder.transform(["Low"])[0])

    raw_rows = []
    score_rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        sampled_cache = {}
        for method in SAMPLING_METHODS:
            sampled_cache[method] = resample_train(
                x_train, y_train, method, high_label=high_label, random_state=SEED + fold
            )

        for method in SAMPLING_METHODS:
            x_fit, y_fit = sampled_cache[method]
            for model_name, factory in MODEL_FACTORIES.items():
                model = factory(SEED + fold)
                model.fit(x_fit, y_fit)
                y_pred = model.predict(x_test)
                scores = score_matrix(model, x_test, labels)
                row = {
                    "Fold": fold,
                    "Model": model_name,
                    "Model_ZH": MODEL_LABELS_ZH[model_name],
                    "Method": method,
                    "Method_ZH": METHOD_LABELS_ZH[method],
                }
                row.update(evaluate(y_test, y_pred, labels, high_label, medium_label, low_label))
                raw_rows.append(row)
                y_test_names = encoder.inverse_transform(y_test)
                for sample_idx, true_label_name in enumerate(y_test_names):
                    score_row = {
                        "Fold": fold,
                        "Model": model_name,
                        "Model_ZH": MODEL_LABELS_ZH[model_name],
                        "Method": method,
                        "Method_ZH": METHOD_LABELS_ZH[method],
                        "True_Label_Name": true_label_name,
                        "True_Label_ZH": CLASS_LABELS_ZH[true_label_name],
                    }
                    for label_idx, class_name in enumerate(encoder.classes_):
                        score_row[f"Score_{class_name}"] = scores[sample_idx, label_idx]
                    score_rows.append(score_row)

    raw_df = pd.DataFrame(raw_rows)
    score_df = pd.DataFrame(score_rows)
    raw_df.to_csv(result_dir / "latex_cv_5fold_raw_results.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(result_dir / "latex_cv_5fold_oof_scores.csv", index=False, encoding="utf-8-sig")

    metric_cols = [
        "High_F1",
        "Medium_F1",
        "Low_F1",
        "High_Precision",
        "High_Recall",
        "Macro_F1",
        "Balanced_Accuracy",
        "G_mean",
    ]
    summary = (
        raw_df.groupby(["Model", "Model_ZH", "Method", "Method_ZH"], as_index=False)[metric_cols]
        .agg(["mean", "std", "var"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.sort_values(["High_F1_mean", "Macro_F1_mean"], ascending=False)
    summary.to_csv(result_dir / "latex_cv_5fold_summary_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(
        result_dir / "latex_cv_5fold_summary_mean_variance_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_latex_table(summary, table_dir / "cv_results_table.tex")
    plot_metric_boxplot(raw_df, "High_F1", "高质量类F1（5折）", figure_dir / "cv_box_high_f1.png")
    plot_metric_boxplot(raw_df, "Macro_F1", "宏平均F1（5折）", figure_dir / "cv_box_macro_f1.png")
    plot_metric_boxplot(raw_df, "G_mean", "G-均值（5折）", figure_dir / "cv_box_gmean.png")
    plot_roc_panel(
        score_df,
        summary,
        ["High", "Medium", "Low"],
        figure_dir / "cv_roc_top_combinations.png",
        result_dir / "latex_cv_roc_auc_top_combinations.csv",
    )

    best = summary.iloc[0]
    second = summary.iloc[1]
    report = [
        "LaTeX论文用5折交叉验证实验结果",
        "=" * 48,
        f"样本总数: {len(df)}",
        f"数值特征数量: {len(FEATURE_COLUMNS)}",
        f"类别分布: {dict((str(k), int(v)) for k, v in Counter(df['teaching_quality_label']).items())}",
        f"最优组合: {best.Model_ZH}+{best.Method_ZH}, 高质量F1={best.High_F1_mean:.3f}, 宏平均F1={best.Macro_F1_mean:.3f}, G-均值={best.G_mean_mean:.3f}",
        f"第二组合: {second.Model_ZH}+{second.Method_ZH}, 高质量F1={second.High_F1_mean:.3f}, 宏平均F1={second.Macro_F1_mean:.3f}, G-均值={second.G_mean_mean:.3f}",
    ]
    (result_dir / "latex_cv_5fold_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"结果表: {result_dir / 'latex_cv_5fold_summary_results.csv'}")
    print(f"LaTeX表格: {table_dir / 'cv_results_table.tex'}")


if __name__ == "__main__":
    main()
