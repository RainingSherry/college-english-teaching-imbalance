import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
# 导入不平衡学习相关的包
# 固定随机种子
np.random.seed(42)

# 加载数据
print("加载数据...")
data_path = '/home/ll/codes/英语教学/二次实验/college_english_teaching_evaluation.csv'
df = pd.read_csv(data_path)

# 数据预处理
print("数据预处理...")
# 去除非数值特征
numeric_columns = [
    'attendance_rate', 'assignment_completion_rate', 'avg_assignment_score',
    'exam_score', 'quiz_avg_score', 'participation_frequency', 'lms_logins',
    'avg_session_duration', 'video_completion_rate', 'self_assessment_score',
    'peer_feedback_score', 'teacher_feedback_score', 'engagement_index'
]

X = df[numeric_columns]
y = df['teaching_quality_label']

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"数据形状: {X.shape}")
print(f"标签分布: {np.bincount(y_encoded)}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7:3 划分训练集和测试集
print("划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

# 定义采样方法
samplers = {
    'No Sampling': None,
    'SMOTE': SMOTE(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42)
}

# 定义分类器
classifiers = {
    '随机森林': RandomForestClassifier(random_state=42),
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '决策树': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier()
}

# 存储结果
results = []
roc_data = {label: {} for label in ['高质量', '中质量', '低质量']}
label_mapping = {'High': '高质量', 'Medium': '中质量', 'Low': '低质量'}

print("开始训练和评估...")

# 为每个组合训练模型
for clf_name, clf in classifiers.items():
    for samp_name, sampler in samplers.items():
        print(f"训练 {clf_name} + {samp_name}...")

        # 应用采样
        if sampler is not None:
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            print(f"  重采样后训练集形状: {X_train_resampled.shape}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # 训练模型
        clf_fitted = clf.fit(X_train_resampled, y_train_resampled)

        # 预测
        y_pred = clf_fitted.predict(X_test)
        y_proba = clf_fitted.predict_proba(X_test)

        # 计算F1分数
        f1_scores = f1_score(y_test, y_pred, average=None)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        # 计算G-均值 (geometric mean of per-class accuracies)
        from sklearn.metrics import accuracy_score
        accuracies = []
        for class_idx in range(len(le.classes_)):
            class_mask = (y_test == class_idx)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                accuracies.append(class_acc)
        g_mean = np.sqrt(np.prod(accuracies)) if accuracies else 0

        # 存储结果
        result_row = {
            'Model': f'{clf_name} + {samp_name}',
            '高质量F1': f"{f1_scores[le.transform(['High'])[0]]:.4f}",
            '中质量F1': f"{f1_scores[le.transform(['Medium'])[0]]:.4f}",
            '低质量F1': f"{f1_scores[le.transform(['Low'])[0]]:.4f}",
            '宏平均F1': f"{f1_macro:.4f}",
            'G-均值': f"{g_mean:.4f}"
        }
        results.append(result_row)

        # 存储ROC数据
        for class_idx, class_name in enumerate(le.classes_):
            chinese_label = label_mapping[class_name]
            if chinese_label not in roc_data:
                roc_data[chinese_label] = {}

            fpr, tpr, _ = roc_curve((y_test == class_idx).astype(int), y_proba[:, class_idx])
            roc_auc = auc(fpr, tpr)

            key = f"{clf_name}_{samp_name}"
            roc_data[chinese_label][key] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc,
                'model': clf_name,
                'sampler': samp_name
            }

# 保存结果到CSV
print("保存结果到CSV...")
results_df = pd.DataFrame(results)
results_df.to_csv('/home/ll/codes/英语教学/二次实验/model_performance_results.csv', index=False, encoding='utf-8-sig')

# 绘制ROC曲线
print("绘制ROC曲线...")
colors = {
    '随机森林': 'blue',
    '逻辑回归': 'red',
    '决策树': 'green',
    'SVM': 'orange',
    'KNN': 'purple'
}

line_styles = {
    'No Sampling': '-',
    'SMOTE': '--',
    'SMOTEENN': '-.',
    'BorderlineSMOTE': ':'
}

# 文件名映射
filename_mapping = {
    '高质量': 'high',
    '中质量': 'medium',
    '低质量': 'low'
}

for class_name in ['高质量', '中质量', '低质量']:
    plt.figure(figsize=(12, 8))

    for key, data in roc_data[class_name].items():
        model_name = data['model']
        sampler_name = data['sampler']

        plt.plot(data['fpr'], data['tpr'],
                color=colors[model_name],
                linestyle=line_styles[sampler_name],
                linewidth=2,
                label=f'{model_name} + {sampler_name} (AUC = {data["auc"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率', fontsize=12)
    plt.ylabel('真正率', fontsize=12)
    plt.title(f'{class_name}标签的ROC曲线', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 保存图像
    filename = filename_mapping[class_name]
    plt.savefig(f'/home/ll/codes/英语教学/二次实验/roc_curve_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 输出到txt文件
print("生成输出文件...")
with open('/home/ll/codes/英语教学/二次实验/experiment_output.txt', 'w', encoding='utf-8') as f:
    f.write("英语教学质量评估分类实验结果\n")
    f.write("=" * 50 + "\n\n")

    f.write("数据信息:\n")
    f.write(f"- 样本总数: {len(df)}\n")
    f.write(f"- 特征数量: {len(numeric_columns)}\n")
    f.write(f"- 训练集大小: {len(X_train)}\n")
    f.write(f"- 测试集大小: {len(X_test)}\n")
    f.write(f"- 类别标签: {list(le.classes_)}\n\n")

    f.write("使用的数值特征:\n")
    for i, feature in enumerate(numeric_columns, 1):
        f.write(f"{i}. {feature}\n")
    f.write("\n")

    f.write("实验设置:\n")
    f.write("- 随机种子: 42\n")
    f.write("- 训练/测试分割比例: 7:3\n")
    f.write("- 采用分层采样\n\n")

    f.write("使用的采样方法:\n")
    for name, sampler in samplers.items():
        f.write(f"- {name}\n")
    f.write("\n")

    f.write("使用的分类器:\n")
    for name, clf in classifiers.items():
        f.write(f"- {name}\n")
    f.write("\n")

    f.write("性能指标说明:\n")
    f.write("- F1分数: 精确率和召回率的调和平均值\n")
    f.write("- 宏平均F1: 各类别F1分数的简单平均\n")
    f.write("- G-均值: 各类别准确率的几何平均\n\n")

    f.write("详细结果已保存至: model_performance_results.csv\n")
    f.write("ROC曲线图已保存至:\n")
    f.write("- roc_curve_high.png (高质量)\n")
    f.write("- roc_curve_medium.png (中质量)\n")
    f.write("- roc_curve_low.png (低质量)\n\n")

    f.write("实验完成!\n")

print("实验完成!所有结果已保存到相应文件中。")
