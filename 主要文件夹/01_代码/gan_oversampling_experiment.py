import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn


SEED = 42
GAN_HIGH_TARGET_COUNT = 100
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def geometric_mean(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    recalls = []
    for label in labels:
        mask = y_true == label
        if np.any(mask):
            recalls.append(accuracy_score(y_true[mask], y_pred[mask]))
    return float(np.prod(recalls) ** (1 / len(recalls))) if recalls else 0.0


def class_f1(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
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
        "高质量F1": class_f1(y_true, y_pred, high_label),
        "中质量F1": class_f1(y_true, y_pred, medium_label),
        "低质量F1": class_f1(y_true, y_pred, low_label),
        "高质量精确率": precision_score(y_true, y_pred, labels=[high_label], average="macro", zero_division=0),
        "高质量召回率": recall_score(y_true, y_pred, labels=[high_label], average="macro", zero_division=0),
        "宏平均F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "平衡准确率": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "G-均值": geometric_mean(y_true, y_pred, labels),
    }


def smote_resample(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    target_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    counts = Counter(y)
    max_count = target_count or max(counts.values())
    X_parts = [X]
    y_parts = [y]

    for label, count in sorted(counts.items()):
        need = max_count - count
        if need <= 0:
            continue

        class_X = X[y == label]
        if len(class_X) < 2:
            sampled = class_X[rng.integers(0, len(class_X), size=need)]
        else:
            n_neighbors = min(6, len(class_X))
            nn_model = NearestNeighbors(n_neighbors=n_neighbors)
            nn_model.fit(class_X)
            neighbor_indices = nn_model.kneighbors(class_X, return_distance=False)
            base_indices = rng.integers(0, len(class_X), size=need)
            synthetic = []
            for base_idx in base_indices:
                candidates = neighbor_indices[base_idx][1:]
                neighbor_idx = int(rng.choice(candidates))
                gap = rng.random()
                synthetic.append(class_X[base_idx] + gap * (class_X[neighbor_idx] - class_X[base_idx]))
            sampled = np.asarray(synthetic)

        X_parts.append(sampled)
        y_parts.append(np.full(need, label, dtype=y.dtype))

    return np.vstack(X_parts), np.concatenate(y_parts)


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
    minority_X: np.ndarray,
    n_generate: int,
    random_state: int,
    epochs: int = 700,
    noise_dim: int = 16,
) -> np.ndarray:
    set_seed(random_state)
    device = torch.device("cpu")
    data = torch.tensor(minority_X, dtype=torch.float32, device=device)
    batch_size = min(32, len(data))

    generator = Generator(noise_dim, minority_X.shape[1]).to(device)
    discriminator = Discriminator(minority_X.shape[1]).to(device)
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

    lower = minority_X.min(axis=0) - 0.25 * minority_X.std(axis=0)
    upper = minority_X.max(axis=0) + 0.25 * minority_X.std(axis=0)
    return np.clip(synthetic, lower, upper)


def gan_resample(
    X: np.ndarray,
    y: np.ndarray,
    high_label: int,
    random_state: int,
    target_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(y)
    target = target_count or min(GAN_HIGH_TARGET_COUNT, max(counts.values()))
    need = target - counts[high_label]
    if need <= 0:
        return X, y
    synthetic_high = train_minority_wgan(X[y == high_label], need, random_state=random_state)
    X_resampled = np.vstack([X, synthetic_high])
    y_resampled = np.concatenate([y, np.full(need, high_label, dtype=y.dtype)])
    return X_resampled, y_resampled


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
    high_label: int,
    random_state: int,
) -> np.ndarray:
    if method == "SMOTE":
        X_fit, y_fit = smote_resample(X_train, y_train, random_state=random_state)
    elif method == "GAN":
        X_fit, y_fit = gan_resample(X_train, y_train, high_label=high_label, random_state=random_state)
    else:
        X_fit, y_fit = X_train, y_train

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_fit, y_fit)
    return model.predict(X_test)


def main() -> None:
    set_seed(SEED)
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent
    data_path = project_dir / "02_数据" / "college_english_teaching_evaluation.csv"
    result_dir = project_dir / "03_结果"
    result_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["teaching_quality_label"])
    labels = label_encoder.transform(label_encoder.classes_)
    high_label = int(label_encoder.transform(["High"])[0])
    medium_label = int(label_encoder.transform(["Medium"])[0])
    low_label = int(label_encoder.transform(["Low"])[0])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    holdout_rows = []
    holdout_predictions = {}
    for method in ["NoSampling", "SMOTE", "GAN"]:
        y_pred = fit_predict(X_train, y_train, X_test, method, high_label, SEED)
        holdout_predictions[method] = y_pred
        row = {"验证方式": "7:3分层留出", "方法": f"逻辑回归 + {method}"}
        row.update(evaluate(y_test, y_pred, labels, high_label, medium_label, low_label))
        holdout_rows.append(row)

    cv_rows = []
    fold_rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for method in ["NoSampling", "SMOTE", "GAN"]:
        per_fold = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            fold_scaler = StandardScaler()
            X_fold_train = fold_scaler.fit_transform(X[train_idx])
            X_fold_test = fold_scaler.transform(X[test_idx])
            y_fold_train = y[train_idx]
            y_fold_test = y[test_idx]
            y_pred = fit_predict(
                X_fold_train,
                y_fold_train,
                X_fold_test,
                method,
                high_label,
                random_state=SEED + fold,
            )
            metrics = evaluate(y_fold_test, y_pred, labels, high_label, medium_label, low_label)
            fold_row = {"fold": fold, "方法": f"逻辑回归 + {method}"}
            fold_row.update(metrics)
            fold_rows.append(fold_row)
            per_fold.append(metrics)

        summary = {"验证方式": "5折分层交叉验证", "方法": f"逻辑回归 + {method}"}
        for metric in [
            "高质量F1",
            "中质量F1",
            "低质量F1",
            "高质量精确率",
            "高质量召回率",
            "宏平均F1",
            "平衡准确率",
            "G-均值",
        ]:
            values = np.asarray([row[metric] for row in per_fold], dtype=float)
            summary[f"{metric}_均值"] = values.mean()
            summary[f"{metric}_标准差"] = values.std(ddof=1)
        cv_rows.append(summary)

    holdout_df = pd.DataFrame(holdout_rows)
    cv_df = pd.DataFrame(cv_rows)
    fold_df = pd.DataFrame(fold_rows)

    holdout_df.to_csv(result_dir / "gan_oversampling_holdout_results.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(result_dir / "stratified_kfold_gan_smote_results.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(result_dir / "stratified_kfold_gan_smote_fold_details.csv", index=False, encoding="utf-8-sig")
    holdout_df.round(3).to_csv(
        result_dir / "论文补充表_GAN_7比3分层留出结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_df.round(3).to_csv(
        result_dir / "论文补充表_GAN_5折分层交叉验证结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paper_summary_rows = []
    for method in ["逻辑回归 + NoSampling", "逻辑回归 + SMOTE", "逻辑回归 + GAN"]:
        holdout_row = holdout_df[holdout_df["方法"] == method].iloc[0]
        cv_row = cv_df[cv_df["方法"] == method].iloc[0]
        paper_summary_rows.append(
            {
                "方法": method,
                "7:3高质量F1": f"{holdout_row['高质量F1']:.3f}",
                "7:3宏平均F1": f"{holdout_row['宏平均F1']:.3f}",
                "7:3 G-均值": f"{holdout_row['G-均值']:.3f}",
                "5折高质量F1": f"{cv_row['高质量F1_均值']:.3f}±{cv_row['高质量F1_标准差']:.3f}",
                "5折宏平均F1": f"{cv_row['宏平均F1_均值']:.3f}±{cv_row['宏平均F1_标准差']:.3f}",
                "5折G-均值": f"{cv_row['G-均值_均值']:.3f}±{cv_row['G-均值_标准差']:.3f}",
            }
        )
    pd.DataFrame(paper_summary_rows).to_csv(
        result_dir / "论文表4_逻辑回归GAN补充分层验证结果.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_original = pd.DataFrame(X_test_raw, columns=FEATURE_COLUMNS)
    y_test_labels = label_encoder.inverse_transform(y_test)
    smote_pred_labels = label_encoder.inverse_transform(holdout_predictions["SMOTE"])
    misclassification_df = test_original.copy()
    misclassification_df["真实标签"] = y_test_labels
    misclassification_df["SMOTE预测标签"] = smote_pred_labels
    misclassification_df["错误类型"] = "其他"
    false_positive_mask = (y_test != high_label) & (holdout_predictions["SMOTE"] == high_label)
    false_negative_mask = (y_test == high_label) & (holdout_predictions["SMOTE"] != high_label)
    true_positive_mask = (y_test == high_label) & (holdout_predictions["SMOTE"] == high_label)
    misclassification_df.loc[false_positive_mask, "错误类型"] = "非高质量误判为高质量"
    misclassification_df.loc[false_negative_mask, "错误类型"] = "高质量漏判"
    misclassification_df.loc[true_positive_mask, "错误类型"] = "高质量正确识别"
    focused_misclassification = misclassification_df[
        misclassification_df["错误类型"].isin(["非高质量误判为高质量", "高质量漏判", "高质量正确识别"])
    ].copy()
    focused_misclassification.to_csv(
        result_dir / "high_quality_smote_misclassification_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    for group_name in ["高质量正确识别", "高质量漏判", "非高质量误判为高质量"]:
        group = focused_misclassification[focused_misclassification["错误类型"] == group_name]
        row = {"样本组": group_name, "样本数": len(group)}
        for feature in FEATURE_COLUMNS:
            row[f"{feature}_均值"] = group[feature].mean() if len(group) else np.nan
        summary_rows.append(row)
    misclassification_summary_df = pd.DataFrame(summary_rows)
    misclassification_summary_df.to_csv(
        result_dir / "high_quality_smote_misclassification_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paper_misclassification_df = misclassification_summary_df[
        [
            "样本组",
            "样本数",
            "exam_score_均值",
            "video_completion_rate_均值",
            "teacher_feedback_score_均值",
            "engagement_index_均值",
        ]
    ].rename(
        columns={
            "exam_score_均值": "考试分数均值",
            "video_completion_rate_均值": "视频完成率均值",
            "teacher_feedback_score_均值": "教师反馈均值",
            "engagement_index_均值": "参与度指数均值",
        }
    )
    paper_misclassification_df.round(3).fillna("--").to_csv(
        result_dir / "论文补充表_SMOTE高质量错分特征分析.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paper_misclassification_df.round(3).fillna("--").to_csv(
        result_dir / "论文表5_逻辑回归SMOTE高质量错分样本特征均值.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report_path = result_dir / "gan_oversampling_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GAN过采样补充实验报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"样本总数: {len(df)}\n")
        f.write(f"数值特征数量: {len(FEATURE_COLUMNS)}\n")
        f.write(f"类别映射: {dict(zip(label_encoder.classes_, [int(label) for label in labels]))}\n")
        f.write(f"总体类别分布: {dict((int(k), int(v)) for k, v in Counter(y).items())}\n")
        f.write(f"7:3训练集类别分布: {dict((int(k), int(v)) for k, v in Counter(y_train).items())}\n")
        f.write(f"7:3测试集类别分布: {dict((int(k), int(v)) for k, v in Counter(y_test).items())}\n\n")
        f.write("7:3分层留出结果:\n")
        f.write(holdout_df.round(4).to_string(index=False))
        f.write("\n\n5折分层交叉验证结果:\n")
        f.write(cv_df.round(4).to_string(index=False))
        f.write("\n\n逻辑回归+SMOTE高质量相关错分分析:\n")
        f.write(f"高质量正确识别: {int(true_positive_mask.sum())}\n")
        f.write(f"高质量漏判: {int(false_negative_mask.sum())}\n")
        f.write(f"非高质量误判为高质量: {int(false_positive_mask.sum())}\n")
        f.write(misclassification_summary_df.round(3).to_string(index=False))
        f.write("\n")

    print(holdout_df.round(4).to_string(index=False))
    print()
    print(cv_df.round(4).to_string(index=False))
    print(f"\n结果已保存至: {report_path}")


if __name__ == "__main__":
    main()
