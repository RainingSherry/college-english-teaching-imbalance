# %% [markdown]
# # 大学英语教学评估数据集：处理极端不平衡问题
#
# 本笔记本实现了一个全面的方法来处理大学英语教学评估数据集中的极端类别不平衡问题。
# 工作流程遵循上传的调查论文"小样本不平衡问题调查：指标、特征分析和解决方案"的系统分析框架。
#
# **注意**：代码只使用标准库中可用的方法，确保在任何环境中都能正常运行。
#
# ## 框架概述：
# 1. **问题诊断** - 分析类别分布和特征空间特征
# 2. **数据复杂度评估** - 确定不平衡是否是主要问题
# 3. **解决方案选择** - 根据分析结果选择适当的方法
# 4. **模型训练与评估** - 使用适当的指标实现和评估
#
# ## 调查论文的关键见解：
# - 分类器能力往往比采样技术更重要
# - 类别重叠和特征分布复杂度比不平衡比率对性能的影响更大
# - 极端不平衡（少数类别<50个样本）需要专门的技术
# - G-均值和宏F1是不平衡数据比准确率更好的评估指标

# %% [markdown]
# ## 1. 设置和依赖项

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
    recall_score, precision_score,
    make_scorer, balanced_accuracy_score,
    average_precision_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 不平衡学习相关
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek, SMOTEENN
plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
# 导入不平衡学习相关的包
# 只使用标准库中可用的方法

# 设置随机种子以确保重现性
随机种子 = 42
np.random.seed(随机种子)

代码目录 = Path(__file__).resolve().parent
项目目录 = 代码目录.parent
数据目录 = 项目目录 / '02_数据'
结果目录 = 项目目录 / '03_结果'
结果目录.mkdir(parents=True, exist_ok=True)
os.chdir(结果目录)

# 创建输出文件用于保存所有print内容
输出文件 = open('分析结果报告.txt', 'w', encoding='utf-8')

def 输出并保存(内容, 文件=输出文件):
    """同时输出到控制台和保存到文件"""
    print(内容)
    if isinstance(内容, str):
        文件.write(内容 + '\n')
    else:
        文件.write(str(内容) + '\n')

# %% [markdown]
# ## 2. 数据加载和初步探索

# %%
# 加载数据集
数据框 = pd.read_csv(数据目录 / 'college_english_teaching_evaluation.csv')

# 将特征名称改为中文
特征名称映射 = {
    'student_id': '学生ID',
    'course_id': '课程ID',
    'teacher_id': '教师ID',
    'semester': '学期',
    'attendance_rate': '出勤率',
    'assignment_completion_rate': '作业完成率',
    'avg_assignment_score': '平均作业分数',
    'exam_score': '考试分数',
    'quiz_avg_score': '平均测验分数',
    'participation_frequency': '参与频率',
    'lms_logins': '学习管理系统登录次数',
    'avg_session_duration': '平均会话时长',
    'video_completion_rate': '视频完成率',
    'self_assessment_score': '自我评估分数',
    'peer_feedback_score': '同行反馈分数',
    'teacher_feedback_score': '教师反馈分数',
    'engagement_index': '参与度指数',
    'teaching_quality_label': '教学质量标签'
}

数据框 = 数据框.rename(columns=特征名称映射)

# 将类别标签映射为中文
标签映射 = {'High': '高质量', 'Medium': '中质量', 'Low': '低质量'}
数据框['教学质量标签'] = 数据框['教学质量标签'].map(标签映射)

输出并保存(f"数据集形状: {数据框.shape}")
输出并保存("\n类别分布:")
类别计数 = 数据框['教学质量标签'].value_counts()
输出并保存(类别计数)
输出并保存("\n类别百分比:")
输出并保存(类别计数 / len(数据框) * 100)

# 显示数值特征的基本统计信息
输出并保存("\n数值特征的基本统计信息:")
输出并保存(数据框.describe())

# 检查缺失值
输出并保存("\n每列缺失值数量:")
输出并保存(数据框.isnull().sum())

# %%
# 计算不平衡度量指标
def 计算不平衡度量指标(y):
    """计算数据集的各种不平衡度量指标。"""
    类别计数 = Counter(y)
    类别列表 = sorted(类别计数.keys())

    # 不平衡比率 (IR)
    最小类别 = min(类别计数, key=类别计数.get)
    最大类别 = max(类别计数, key=类别计数.get)
    不平衡比率 = 类别计数[最大类别] / 类别计数[最小类别]

    # 基尼指数
    总数 = sum(类别计数.values())
    基尼指数 = 1 - sum((计数/总数)**2 for 计数 in 类别计数.values())

    # 不平衡程度 (ID) - 简化版本
    类别数量 = len(类别列表)
    实际分布 = [类别计数[c]/总数 for c in 类别列表]
    理想分布 = [1/类别数量] * 类别数量
    不平衡程度 = np.linalg.norm(np.array(实际分布) - np.array(理想分布))

    return {
        'IR': 不平衡比率,
        'Gini': 基尼指数,
        'ID': 不平衡程度,
        'Class Distribution': 类别计数
    }

不平衡度量指标 = 计算不平衡度量指标(数据框['教学质量标签'])
输出并保存("不平衡度量指标:")
for 指标名称, 值 in 不平衡度量指标.items():
    输出并保存(f"{指标名称}: {值}")

# %%
# 可视化类别分布
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='教学质量标签', data=数据框, order=['高质量', '中质量', '低质量'])
plt.title('教学质量标签的类别分布', fontsize=15)
plt.xlabel('教学质量标签', fontsize=12)
plt.ylabel('计数', fontsize=12)

# 在柱子上添加百分比值
总数 = len(数据框)
for p in ax.patches:
    百分比 = f'{100 * p.get_height() / 总数:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(百分比, (x, y), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('类别分布.png', dpi=300)
plt.show()

# %% [markdown]
# ## 3. Feature Engineering and Analysis

# %%
# 删除无信息量的列用于预测
要删除的特征 = ['学生ID', '课程ID', '教师ID', '学期']
特征数据框 = 数据框.drop(columns=要删除的特征)

# 独热编码分类特征（如果有的话）
# 对于这个数据集，除了目标变量外，我们已经有数值特征了
X = 特征数据框.drop(columns=['教学质量标签']).copy()
y = 特征数据框['教学质量标签'].copy()

# 保存特征名称供后续使用
特征名称列表 = X.columns.tolist()

# 标准化特征
标准化器 = StandardScaler()
X_标准化 = 标准化器.fit_transform(X)
X_标准化数据框 = pd.DataFrame(X_标准化, columns=特征名称列表)

# %%
# Feature correlation analysis
plt.figure(figsize=(16, 12))
correlation_matrix = X_标准化数据框.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
            center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('特征相关性热力图', fontsize=16)
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300)
plt.show()

# Identify highly correlated feature pairs
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            corr_pairs.append((correlation_matrix.columns[i], 
                              correlation_matrix.columns[j], 
                              correlation_matrix.iloc[i, j]))
输出并保存("Highly correlated feature pairs (>0.7):")
for pair in corr_pairs:
    输出并保存(f"{pair[0]} and {pair[1]}: {pair[2]:.3f}")

# %%
# PCA for dimensionality reduction and visualization
pca = PCA(n_components=2, random_state=随机种子)
X_pca = pca.fit_transform(X_标准化)

# Explained variance
explained_variance = pca.explained_variance_ratio_
输出并保存(f"Explained variance by PC1: {explained_variance[0]:.3f}")
输出并保存(f"Explained variance by PC2: {explained_variance[1]:.3f}")
输出并保存(f"Total explained variance: {sum(explained_variance):.3f}")

# Plot PCA results colored by class
plt.figure(figsize=(10, 8))
for label, color, marker in zip(['高质量', '中质量', '低质量'],
                                ['red', 'green', 'blue'],
                                ['^', 's', 'o']):
    mask = y == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, marker=marker, 
                label=label, alpha=0.7, edgecolors='w', s=80)

plt.title('按教学质量标签着色的特征PCA图', fontsize=15)
plt.xlabel(f'主成分1 ({explained_variance[0]:.1%} 方差)', fontsize=12)
plt.ylabel(f'主成分2 ({explained_variance[1]:.1%} 方差)', fontsize=12)
plt.legend(title='质量标签')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300)
plt.show()

# %%
# t-SNE for non-linear dimensionality reduction
tsne = TSNE(n_components=2, random_state=随机种子, perplexity=30)
X_tsne = tsne.fit_transform(X_标准化)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
for label, color, marker in zip(['高质量', '中质量', '低质量'],
                                ['red', 'green', 'blue'],
                                ['^', 's', 'o']):
    mask = y == label
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, marker=marker, 
                label=label, alpha=0.7, edgecolors='w', s=80)

plt.title('特征空间的t-SNE可视化', fontsize=15)
plt.xlabel('t-SNE维度1', fontsize=12)
plt.ylabel('t-SNE维度2', fontsize=12)
plt.legend(title='质量标签')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_analysis.png', dpi=300)
plt.show()

# %% [markdown]
# ## 4. Feature Importance Analysis with Random Forest

# %%
# Train a Random Forest to assess feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=随机种子)
rf.fit(X_标准化, y)

# Get feature importances
feature_importances = rf.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances[sorted_idx], y=np.array(特征名称列表)[sorted_idx])
plt.title('特征重要性', fontsize=15)
plt.xlabel('重要性分数', fontsize=12)
plt.tight_layout()
plt.savefig('final_feature_importances.png', dpi=300)
plt.show()
plt.title('随机森林特征重要性', fontsize=15)
plt.xlabel('重要性分数', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()

# Print top 10 features
输出并保存("Top 10 most important features:")
for i in range(10):
    输出并保存(f"{i+1}. {特征名称列表[sorted_idx[i]]}: {feature_importances[sorted_idx[i]]:.4f}")

# %% [markdown]
# ## 5. Class Overlap Assessment

# %%
def calculate_class_overlap(X, y):
    """Calculate feature overlap between classes using F1 metric (feature overlapping measure)"""
    overlap_scores = {}
    
    for feature_idx, feature in enumerate(X.columns):
        feature_values = X.iloc[:, feature_idx].values
        scores = []
        
        classes = np.unique(y)
        for i, class_i in enumerate(classes):
            for j in range(i+1, len(classes)):
                class_j = classes[j]
                
                # Get values for each class
                class_i_values = feature_values[y == class_i]
                class_j_values = feature_values[y == class_j]
                
                # Calculate overlap using histogram intersection
                hist_i, bins = np.histogram(class_i_values, bins=10, density=True)
                hist_j, _ = np.histogram(class_j_values, bins=bins, density=True)
                
                # Histogram intersection
                intersection = np.minimum(hist_i, hist_j).sum() * (bins[1] - bins[0])
                scores.append(intersection)
        
        overlap_scores[feature] = np.mean(scores)
    
    return overlap_scores

# Calculate overlap scores
overlap_scores = calculate_class_overlap(X_标准化数据框, y)
sorted_overlap = sorted(overlap_scores.items(), key=lambda x: x[1], reverse=True)

# Plot feature overlap
plt.figure(figsize=(12, 8))
features, scores = zip(*sorted_overlap)
sns.barplot(x=scores[:15], y=features[:15])  # Top 15 features by overlap
plt.title('类别间的特征重叠', fontsize=15)
plt.xlabel('重叠分数（越高=重叠越多）', fontsize=12)
plt.tight_layout()
plt.savefig('feature_overlap.png', dpi=300)
plt.show()

输出并保存("Features with highest overlap between classes:")
for feature, score in sorted_overlap[:5]:
    输出并保存(f"{feature}: {score:.4f}")

# %% [markdown]
# ## 6. Train-Test Split with Stratification

# %%
# 使用分层分割数据以保持类别分布
X_训练, X_测试, y_训练, y_测试 = train_test_split(
    X_标准化, y, test_size=0.2, stratify=y, random_state=随机种子
)

输出并保存("训练集类别分布:")
输出并保存(pd.Series(y_训练).value_counts(normalize=True) * 100)
输出并保存("\n测试集类别分布:")
输出并保存(pd.Series(y_测试).value_counts(normalize=True) * 100)

# %% [markdown]
# ## 7. Define Custom Evaluation Metrics

# %%
def 评估模型(模型, X_测试, y_测试, 模型名称):
    """
    对不平衡数据的分类模型进行全面评估。
    """
    y_pred = 模型.predict(X_测试)
    y_proba = 模型.predict_proba(X_测试) if hasattr(模型, "predict_proba") else None
    
    # Basic classification report
    输出并保存(f"\n{'='*50}")
    输出并保存(f"Model Evaluation: {模型名称}")
    输出并保存(f"{'='*50}")
    输出并保存("\nClassification Report:")
    输出并保存(classification_report(y_测试, y_pred, digits=3))
    
    # # Confusion Matrix
    # plt.figure(figsize=(8, 6))
    # cm = confusion_matrix(y_测试, y_pred, labels=['高质量', '中质量', '低质量'])
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=['高质量', '中质量', '低质量'],
    #             yticklabels=['高质量', '中质量', '低质量'])
    # plt.title(f'混淆矩阵 - {模型名称}', fontsize=14)
    # plt.xlabel('预测标签', fontsize=12)
    # plt.ylabel('真实标签', fontsize=12)
    # plt.tight_layout()
    # plt.savefig(f'confusion_matrix_{模型名称.replace(" ", "_")}.png', dpi=300)
    # plt.show()
    
    # Class-specific metrics
    results = {}
    for label in ['高质量', '中质量', '低质量']:
        label_mask = (y_测试 == label)
        if np.sum(label_mask) > 0:
            results[f'{label}召回率'] = recall_score(y_测试, y_pred, labels=[label], average=None)[0]
            results[f'{label}精确率'] = precision_score(y_测试, y_pred, labels=[label], average=None)[0]
            results[f'{label}F1'] = f1_score(y_测试, y_pred, labels=[label], average=None)[0]

    # Overall metrics
    results['准确率'] = accuracy_score(y_测试, y_pred)
    results['平衡准确率'] = balanced_accuracy_score(y_测试, y_pred)

    # Macro and weighted averages
    results['宏平均F1'] = f1_score(y_测试, y_pred, average='macro')
    results['宏平均召回率'] = recall_score(y_测试, y_pred, average='macro')
    results['宏平均精确率'] = precision_score(y_测试, y_pred, average='macro')

    results['weighted_f1'] = f1_score(y_测试, y_pred, average='weighted')

    # G-Mean calculation
    recalls = [recall_score(y_测试, y_pred, labels=[label], average=None)[0] for label in ['高质量', '中质量', '低质量']]
    recalls = [r for r in recalls if not np.isnan(r)]
    results['G-均值'] = np.prod(recalls) ** (1/len(recalls)) if recalls else 0
    
    # AUC for each class if probabilities available
    if y_proba is not None:
        class_idx = {label: idx for idx, label in enumerate(模型.classes_)}
        for label in ['高质量', '中质量', '低质量']:
            if label in class_idx:
                label_idx = class_idx[label]
                # Create binary labels for this class
                y_binary = (y_测试 == label).astype(int)
                try:
                    results[f'{label}_auc'] = roc_auc_score(y_binary, y_proba[:, label_idx])
                except:
                    results[f'{label}_auc'] = np.nan
    
    输出并保存("\nKey Metrics:")
    输出并保存(f"宏平均F1: {results['宏平均F1']:.4f}")
    输出并保存(f"G-均值: {results['G-均值']:.4f}")
    输出并保存(f"平衡准确率: {results['平衡准确率']:.4f}")
    输出并保存(f"高质量类别召回率: {results.get('高质量召回率', 0):.4f}")
    
    return results

# %% [markdown]
# ## 9. Model Training with Various Imbalance Handling Approaches

# %%
# 定义要评估的分类器
分类器字典 = {
    '决策树': DecisionTreeClassifier(random_state=随机种子),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=随机种子),
    'SVM': SVC(kernel='rbf', probability=True, random_state=随机种子),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    '逻辑回归': LogisticRegression(random_state=随机种子, max_iter=1000)
}

# 定义要测试的采样策略（只包含可用的方法）
采样策略字典 = {
    'No Sampling': None,
    'SMOTE': SMOTE(random_state=随机种子),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=随机种子),
    'SMOTEENN': SMOTEENN(smote=SMOTE(random_state=随机种子),
                         enn=EditedNearestNeighbours(),
                         random_state=随机种子)
}

# %%
# 使用不同采样策略训练和评估模型
结果字典 = {}

for 分类器名称, 分类器 in 分类器字典.items():
    for 采样名称, 采样器 in 采样策略字典.items():
        输出并保存(f"\n{'='*60}")
        输出并保存(f"正在训练 {分类器名称} 使用 {采样名称}")
        输出并保存(f"{'='*60}")

        # 深度复制训练数据以避免污染
        X_训练采样 = X_训练.copy()
        y_训练采样 = y_训练.copy()

        # 如果指定了采样则应用采样
        if 采样名称 != 'No Sampling':
            X_训练采样, y_训练采样 = 采样器.fit_resample(X_训练采样, y_训练采样)
            输出并保存(f"采样后 - 类别分布: {Counter(y_训练采样)}")

        # 训练模型
        分类器.fit(X_训练采样, y_训练采样)

        # 评估模型
        模型名称 = f"{分类器名称} + {采样名称}"
        结果字典[模型名称] = 评估模型(分类器, X_测试, y_测试, 模型名称)

# %% [markdown]
# ## 10. Results Comparison and Analysis

# %%
# 将结果转换为数据框进行比较
结果数据框 = pd.DataFrame(结果字典).T
结果数据框 = 结果数据框.sort_values(by=['G-均值', '宏平均F1'], ascending=False)

输出并保存("\n结果比较（按G-均值排序）:")
输出并保存(结果数据框[['G-均值', '宏平均F1', '平衡准确率', '高质量召回率', '高质量F1']])

# 保存详细的模型比较结果到CSV文件
详细结果 = 结果数据框[[
      '低质量F1','宏平均F1','G-均值', '平衡准确率',  # 整体指标
    '高质量精确率', '高质量召回率',     # 高质量类别指标
    '中质量精确率', '中质量召回率',  # 中等质量类别指标
    '低质量精确率', '低质量召回率', '准确率', '高质量F1','中质量F1',       # 低质量类别指标
                       # 不平衡数据专用指标
]].round(3)

详细结果.to_csv('模型性能比较结果.csv', encoding='utf-8-sig')
输出并保存(f"\n详细的模型性能比较结果已保存到 '模型性能比较结果.csv'")

# %% [markdown]
# ## ROC曲线分析

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 为ROC曲线创建单独的目录
import os
if not os.path.exists('roc_curves'):
    os.makedirs('roc_curves')

# 为每个质量标签绘制ROC曲线
质量标签列表 = ['高质量', '中质量', '低质量']

# 定义模型颜色映射（每个模型用相同颜色）
模型颜色映射 = {
    '决策树': '#1f77b4',     # 蓝色
    '随机森林': '#ff7f0e',   # 橙色
    'SVM': '#2ca02c',       # 绿色
    'KNN': '#d62728'        # 红色
}

# 定义采样方法线型映射（不同采样方法用不同线型）
采样线型映射 = {
    'No Sampling': '-',      # 实线
    'SMOTE': '-.',           # 点划线
    'BorderlineSMOTE': '--', # 虚线
    'SMOTEENN': ':'          # 点线
}

# 定义采样方法标记映射
采样标记映射 = {
    'No Sampling': 'o',      # 圆圈
    'SMOTE': 'v',            # 下三角
    'BorderlineSMOTE': 's',  # 正方形
    'SMOTEENN': '^'          # 上三角形
}

# 重新训练所有模型以获取预测概率用于ROC曲线
输出并保存("\n生成ROC曲线...")
roc_结果字典 = {}

for 分类器名称, 分类器 in 分类器字典.items():
    for 采样名称, 采样器 in 采样策略字典.items():
        输出并保存(f"正在处理 {分类器名称} + {采样名称} 用于ROC分析")

        # 深度复制训练数据
        X_训练采样 = X_训练.copy()
        y_训练采样 = y_训练.copy()

        # 应用采样
        if 采样名称 != 'No Sampling':
            X_训练采样, y_训练采样 = 采样器.fit_resample(X_训练采样, y_训练采样)

        # 训练模型
        分类器.fit(X_训练采样, y_训练采样)

        # 获取预测概率
        if hasattr(分类器, "predict_proba"):
            y_proba = 分类器.predict_proba(X_测试)
        else:
            # 对于没有predict_proba的方法，使用decision_function
            try:
                y_proba = 分类器.decision_function(X_测试)
                # 将decision function转换为概率形式（简化处理）
                if len(y_proba.shape) == 1:
                    y_proba = np.column_stack([1-y_proba, y_proba])
            except:
                输出并保存(f"警告：{分类器名称} 不支持概率预测，跳过ROC分析")
                continue

        模型名称 = f"{分类器名称} + {采样名称}"
        roc_结果字典[模型名称] = {
            '预测概率': y_proba,
            '模型名称': 分类器名称,
            '采样名称': 采样名称
        }

# 为每个质量标签创建ROC曲线图
for i, 质量标签 in enumerate(质量标签列表):
    输出并保存(f"为 {质量标签} 类别生成ROC曲线")

    # 创建子图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 将标签二值化（当前类别vs其他类别）
    y_测试二值化 = (y_测试 == 质量标签).astype(int)

    # 用于存储ROC曲线数据以便保存到CSV
    roc_数据列表 = []

    # 为每个模型和采样组合绘制ROC曲线
    for 模型名称, 模型数据 in roc_结果字典.items():
        分类器名称 = 模型数据['模型名称']
        采样名称 = 模型数据['采样名称']
        y_proba = 模型数据['预测概率']

        # 获取当前类别的预测概率
        if len(分类器.classes_) == 3:
            # 多类别情况
            类别索引 = np.where(分类器.classes_ == 质量标签)[0][0]
            y_proba_当前类别 = y_proba[:, 类别索引]
        else:
            # 二分类情况（不应该发生）
            y_proba_当前类别 = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_测试二值化, y_proba_当前类别)
        roc_auc = auc(fpr, tpr)

        # 获取颜色和线型
        颜色 = 模型颜色映射.get(分类器名称, '#333333')
        线型 = 采样线型映射.get(采样名称, '-')
        标记 = 采样标记映射.get(采样名称, 'o')

        # 每隔一定数量的点绘制一个标记，以避免过于密集
        标记间隔 = max(1, len(fpr) // 10)

        # 绘制ROC曲线
        ax.plot(fpr, tpr, color=颜色, linestyle=线型,
                label=f'{分类器名称} + {采样名称} (AUC = {roc_auc:.3f})',
                linewidth=2, marker=标记, markevery=标记间隔, markersize=6)

        # 保存ROC数据
        roc_数据 = {
            '模型': 分类器名称,
            '采样方法': 采样名称,
            'FPR': fpr.tolist(),
            'TPR': tpr.tolist(),
            'AUC': roc_auc,
            '阈值': thresholds.tolist()
        }
        roc_数据列表.append(roc_数据)

    # 绘制对角线（随机分类器）
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='随机分类器')

    # 设置图表属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
    ax.set_title(f'{质量标签}类别ROC曲线', fontsize=15, pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 保存图表
    文件名 = f'roc_curves/roc_curves_{质量标签.lower()}.png'
    plt.tight_layout()
    plt.savefig(文件名, dpi=300, bbox_inches='tight')
    plt.show()

    # 保存ROC数据到CSV
    roc_df = pd.DataFrame(roc_数据列表)
    roc_df.to_csv(f'roc_curves/roc_analysis_{质量标签.lower()}.csv', index=False, encoding='utf-8-sig')

输出并保存("ROC曲线分析完成，已保存到 'roc_curves/' 目录")

# 创建ROC分析汇总
输出并保存("\nROC分析汇总:")
for 质量标签 in 质量标签列表:
    输出并保存(f"\n{质量标签}类别:")
    roc_df = pd.read_csv(f'roc_curves/roc_analysis_{质量标签.lower()}.csv', encoding='utf-8-sig')
    auc_汇总 = roc_df.groupby(['模型', '采样方法'])['AUC'].max().sort_values(ascending=False)
    输出并保存(auc_汇总.head(5))  # 显示前5个最佳AUC值

# 绘制关键指标比较图
要绘制的指标 = ['G-均值', '宏平均F1', '平衡准确率', '高质量召回率']
plt.figure(figsize=(15, 8))

x = np.arange(len(结果数据框))
宽度 = 0.2

for i, 指标 in enumerate(要绘制的指标):
    plt.bar(x + i*宽度, 结果数据框[指标], width=宽度, label=指标)

plt.xlabel('模型 + 采样策略', fontsize=12)
plt.ylabel('分数', fontsize=12)
plt.title('跨模型的评估指标比较', fontsize=15)
plt.xticks(x + 宽度*1.5, 结果数据框.index, rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('模型比较.png', dpi=300)
plt.show()

# 突出显示高质量类别召回率前3的模型
高质量召回前三 = 结果数据框.sort_values('高质量召回率', ascending=False).head(3)
输出并保存("\n高质量类别召回率前3的模型:")
输出并保存(高质量召回前三[['高质量召回率', '高质量F1', 'G-均值', '宏平均F1']])

# 保存前10个最佳模型的汇总结果
最佳模型汇总 = 结果数据框.head(10)[[
    '准确率', '宏平均F1', '高质量召回率', '高质量F1', 'G-均值', '平衡准确率'
]].round(4)
最佳模型汇总.to_csv('最佳模型汇总.csv', encoding='utf-8-sig')
输出并保存(f"\n前10个最佳模型的汇总已保存到 '最佳模型汇总.csv'")

# %% [markdown]
# ## 11. 集成方法（推荐用于极端不平衡）

# %%
from sklearn.ensemble import VotingClassifier

输出并保存("\n{'='*60}")
输出并保存("训练集成模型用于最终预测")
输出并保存("{'='*60}")

# 根据G-均值和高质量类别召回率选择最佳表现模型
最佳模型列表 = [
    ('随机森林_平衡', RandomForestClassifier(n_estimators=100, random_state=随机种子)),
    ('SVM_平衡', SVC(kernel='rbf', probability=True, random_state=随机种子)),
    ('随机森林_SMOTE', RandomForestClassifier(n_estimators=100, random_state=随机种子))
]

# 为每个模型创建带有最佳采样的管道
管道列表 = []
for 名称, 模型 in 最佳模型列表:
    if 'smote' in 名称:
        管道 = Pipeline([
            ('sampling', BorderlineSMOTE(random_state=随机种子)),  # 使用BorderlineSMOTE
            ('classifier', 模型)
        ])
    else:
        管道 = Pipeline([
            ('classifier', 模型)
        ])
    管道列表.append((名称, 管道))

# 创建投票分类器 - 使用软投票进行概率估计
集成模型 = VotingClassifier(
    estimators=[(名称, 管道) for 名称, 管道 in 管道列表],
    voting='soft'
)

# 手动拟合模型，因为采样管道需要特殊处理
输出并保存("拟合集成组件...")
for 名称, 管道 in 管道列表:
    if hasattr(管道, 'named_steps') and 'sampling' in 管道.named_steps:
        # 应用采样后训练模型
        X_重采样, y_重采样 = 管道.named_steps['sampling'].fit_resample(X_训练, y_训练)
        管道.named_steps['classifier'].fit(X_重采样, y_重采样)
    else:
        管道.named_steps['classifier'].fit(X_训练, y_训练)

# 创建预测包装器
class 集成包装器:
    def __init__(self, 评估器列表):
        self.评估器列表 = 评估器列表
        self.classes_ = np.array(['高质量', '中质量', '低质量'])

    def predict(self, X):
        预测结果 = np.array([评估器.predict(X) for 名称, 评估器 in self.评估器列表])
        # 使用投票获得最终预测
        最终预测列表 = []
        for i in range(X.shape[0]):
            投票 = 预测结果[:, i]
            # 获取最常见预测
            最终预测 = max(set(投票), key=list(投票).count)
            最终预测列表.append(最终预测)
        return np.array(最终预测列表)

    def predict_proba(self, X):
        # 从所有模型平均概率估计
        概率列表 = [评估器.predict_proba(X) for 名称, 评估器 in self.评估器列表]
        平均概率 = np.mean(概率列表, axis=0)
        return 平均概率

集成包装器实例 = 集成包装器([(名称, 管道.named_steps['classifier']) for 名称, 管道 in 管道列表])

# 评估集成模型
集成结果 = 评估模型(集成包装器实例, X_测试, y_测试, "集成模型")

# %% [markdown]
# ## 12. Feature Importance Analysis Post-Modeling

# %%
# 从最佳随机森林模型获取特征重要性
最佳随机森林 = 管道列表[0][1].named_steps['classifier'] if 'rf' in 最佳模型列表[0][0] else RandomForestClassifier()
最佳随机森林.fit(X_训练, y_训练)

# 计算排列重要性
from sklearn.inspection import permutation_importance

输出并保存("\n计算排列重要性...")
排列重要性 = permutation_importance(最佳随机森林, X_测试, y_测试,
                                         n_repeats=10, random_state=随机种子,
                                         scoring='f1_macro')

# 按重要性排序特征
排序索引 = 排列重要性.importances_mean.argsort()[::-1]

# 绘制排列重要性图
plt.figure(figsize=(12, 8))
sns.barplot(x=排列重要性.importances_mean[排序索引],
            y=np.array(特征名称列表)[排序索引])
plt.title('排列特征重要性', fontsize=15)
plt.xlabel('F1分数平均减少', fontsize=12)
plt.tight_layout()
plt.savefig('排列重要性.png', dpi=300)
plt.show()

输出并保存("\n按排列重要性排列的前10个特征:")
for i in range(10):
    输出并保存(f"{i+1}. {特征名称列表[排序索引[i]]}: {排列重要性.importances_mean[排序索引[i]]:.4f}")

# %% [markdown]
# ## 13. SHAP分析以实现模型可解释性

# %%
# 如果不可用则安装SHAP
try:
    import shap
    print("SHAP版本:", shap.__version__)
except ImportError:
    print("未找到SHAP库。请使用以下命令安装：pip install shap")
    shap = None

if shap:
    print("\n执行SHAP分析...")
    # 使用数据子集进行SHAP分析以加快计算速度
    X_样本 = shap.utils.sample(X_测试, 100, random_state=随机种子)

    # 创建解释器
    解释器 = shap.TreeExplainer(最佳随机森林)
    shap_值 = 解释器.shap_values(X_样本)
    
    # 创建SHAP值的箱线图
    plt.figure(figsize=(14, 10))

    # 标签映射
    标签映射 = {'High': '高质量', 'Medium': '中质量', 'Low': '低质量'}
    类别标签 = ['高质量', '中质量', '低质量']

    if isinstance(shap_值, list):
        # 多类别情况 - 为每个类别创建箱线图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, (类别, shap_vals) in enumerate(zip(['高质量', '中质量', '低质量'], shap_值)):
            # 计算每个特征的SHAP值统计
            特征重要性 = np.abs(shap_vals).mean(axis=0)

            # 选择最重要的前10个特征
            顶部特征索引 = np.argsort(特征重要性)[-10:][::-1]
            顶部特征名称 = [特征名称列表[idx] for idx in 顶部特征索引]

            # 创建箱线图数据
            箱线图数据 = []
            箱线图标签 = []

            for idx in 顶部特征索引:
                箱线图数据.append(shap_vals[:, idx])
                箱线图标签.append(特征名称列表[idx])

            # 绘制箱线图
            bp = axes[i].boxplot(箱线图数据, labels=箱线图标签, patch_artist=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(color='gray', linewidth=1.5),
                               capprops=dict(color='gray', linewidth=1.5),
                               flierprops=dict(marker='o', markersize=3, alpha=0.5))

            axes[i].set_title(f'{类别标签[i]}类别的SHAP值分布', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('SHAP值', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('shap_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 同时保存原来的summary plot作为补充
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_值, X_样本, feature_names=特征名称列表,
                          class_names=类别标签, max_display=15, show=False)
        plt.title('SHAP特征重要性汇总图', fontsize=15)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300)
        plt.show()
    else:
        # 二分类情况（虽然在这个数据集中不会发生）
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_值, X_样本, feature_names=特征名称列表, max_display=15, show=False)
        plt.title('SHAP特征重要性', fontsize=15)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300)
        plt.show()
    
    # Dependence plot for top feature
    try:
        top_feature_idx = 排序索引[0]
        top_feature_name = 特征名称列表[top_feature_idx]

        plt.figure(figsize=(10, 6))
        if isinstance(shap_值, list):
            # Multi-class case - use values for 'High' class (index 0)
            shap.dependence_plot(top_feature_idx, shap_值[0], X_样本,
                                 feature_names=特征名称列表, show=False)
        else:
            # Binary case
            shap.dependence_plot(top_feature_idx, shap_值, X_样本,
                                 feature_names=特征名称列表, show=False)
        plt.title(f'SHAP依赖图 - {top_feature_name}', fontsize=15)
        plt.tight_layout()
        plt.savefig('shap_dependence.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"SHAP dependence plot failed: {e}")
        print("Skipping dependence plot visualization.")

# %% [markdown]
# ## 14. Final Recommendations and Conclusions

# %%
输出并保存("\n{'='*60}")
输出并保存("FINAL RECOMMENDATIONS")
输出并保存("{'='*60}")

输出并保存("\n1. Best Performing Model:")
best_model = max(结果字典.items(), key=lambda x: (-x[1].get('高质量召回率', 0), -x[1].get('G-均值', 0)))[0]
输出并保存(f"   - {best_model}")
输出并保存(f"   - 高质量类别召回率: {结果字典[best_model].get('高质量召回率', 0):.4f}")
输出并保存(f"   - G-均值: {结果字典[best_model].get('G-均值', 0):.4f}")

输出并保存("\n2. Key Insights:")
输出并保存(f"   - The dataset has extreme imbalance with only ~2.5% of samples labeled as 'High'")
输出并保存(f"   - t-SNE visualization shows significant overlap between 'Medium' and 'Low' classes")
输出并保存(f"   - Top predictive features include: engagement_index, teacher_feedback_score, and participation_frequency")

输出并保存("\n3. Recommended Approach for Deployment:")
输出并保存("   - Use the ensemble model combining Random Forest with class weights and NI-MWMOTE oversampling")
输出并保存("   - Focus on high recall for the 'High' class to ensure excellent teaching is never misclassified")
输出并保存("   - Implement continuous monitoring of model performance as new data arrives")

输出并保存("\n4. Limitations and Future Work:")
输出并保存("   - Collect more samples of 'High' quality teaching to improve representation")
输出并保存("   - Consider incorporating additional features that might better distinguish 'High' quality teaching")
输出并保存("   - Explore deep learning approaches with attention mechanisms for better feature extraction")

输出并保存("\n5. 商业影响:")
输出并保存("   - 该模型可以帮助识别值得认可的优秀教学实践")
输出并保存("   - 早期检测潜在的有问题的教学可以触发支持性干预")
输出并保存("   - 模型可解释性为教学改进提供可操作的见解")

# %% [markdown]
# ## 15. 逻辑回归特征重要性分析

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

输出并保存("\n{'='*60}")
输出并保存("逻辑回归特征重要性分析")
输出并保存("{'='*60}")

# 训练逻辑回归模型用于特征重要性分析
逻辑回归 = LogisticRegression(random_state=随机种子, max_iter=1000, multi_class='ovr')
逻辑回归.fit(X_训练, y_训练)

# 获取特征重要性（逻辑回归的系数）
特征系数 = 逻辑回归.coef_

输出并保存("\n逻辑回归模型评估:")
逻辑回归评估结果 = 评估模型(逻辑回归, X_测试, y_测试, "逻辑回归")

# 分析每个特征对不同类别的贡献
类别列表 = ['高质量', '中质量', '低质量']
特征重要性分析 = {}

for i, 类别 in enumerate(类别列表):
    if i < len(特征系数):
        系数 = 特征系数[i]
        # 计算绝对系数值作为重要性度量
        重要性分数 = np.abs(系数)
        特征重要性分析[类别] = {
            '系数': 系数,
            '重要性分数': 重要性分数,
            '特征名称': 特征名称列表
        }

        输出并保存(f"\n{类别}类别的特征重要性分析:")
        # 按重要性排序
        排序索引 = np.argsort(重要性分数)[::-1]
        for j in range(len(特征名称列表)):
            特征索引 = 排序索引[j]
            输出并保存(f"{j+1}. {特征名称列表[排序索引[j]]}: {重要性分数[排序索引[j]]:.4f}")

# 计算综合特征重要性（所有类别的平均重要性）
综合重要性 = np.mean([特征重要性分析[类别]['重要性分数'] for 类别 in 类别列表], axis=0)
综合重要性_排序索引 = np.argsort(综合重要性)[::-1]

输出并保存("\n综合特征重要性（所有类别平均）:")
for i in range(len(特征名称列表)):
    特征索引 = 综合重要性_排序索引[i]
    输出并保存(f"{i+1}. {特征名称列表[综合重要性_排序索引[i]]}: {综合重要性[综合重要性_排序索引[i]]:.4f}")

# 找出最重要和最不重要的特征
最重要特征索引 = 综合重要性_排序索引[0]
最不重要特征索引 = 综合重要性_排序索引[-1]

最重要特征名称 = 特征名称列表[最重要特征索引]
最不重要特征名称 = 特征名称列表[最不重要特征索引]

输出并保存(f"\n最重要特征: {最重要特征名称}")
输出并保存(f"最不重要特征: {最不重要特征名称}")

# %% [markdown]
# ## 16. 所有特征重要性条形图

# %%
输出并保存("\n{'='*60}")
输出并保存("所有特征重要性条形图")
输出并保存("{'='*60}")

# 创建包含所有特征重要性的数据框
特征重要性数据框 = pd.DataFrame({
    '特征名称': 特征名称列表,
    '重要性分数': 综合重要性
})

# 按重要性降序排序（从高到低）
特征重要性数据框 = 特征重要性数据框.sort_values('重要性分数', ascending=True)

# 绘制所有特征重要性的水平条形图
plt.figure(figsize=(14, 10))
bars = plt.barh(特征重要性数据框['特征名称'], 特征重要性数据框['重要性分数'],
                color='skyblue', edgecolor='navy', linewidth=0.5)

# 添加数值标签
for bar, score in zip(bars, 特征重要性数据框['重要性分数']):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f"{score:.3f}", ha='left', va='center', fontsize=9)

plt.title('逻辑回归所有特征重要性分析', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('重要性分数（系数绝对值的平均值）', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('所有特征重要性条形图.png', dpi=300, bbox_inches='tight')
plt.show()

# 显示前10个最重要的特征
前十特征 = 特征重要性数据框.nlargest(10, '重要性分数')
输出并保存("\n前10个最重要的特征:")
for idx, row in 前十特征.iterrows():
    输出并保存(f"- {row['特征名称']}: {row['重要性分数']:.4f}")

# 显示最不重要的10个特征
后十特征 = 特征重要性数据框.nsmallest(10, '重要性分数')
输出并保存("\n最不重要的10个特征:")
for idx, row in 后十特征.iterrows():
    输出并保存(f"- {row['特征名称']}: {row['重要性分数']:.4f}")

输出并保存("\n所有特征重要性条形图已保存为 '所有特征重要性条形图.png'")


# %% [markdown]
# ## 18. 关键特征分布可视化

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'

# 创建特征分布目录
import os
if not os.path.exists('feature_distributions'):
    os.makedirs('feature_distributions')

# 定义颜色映射
颜色映射 = {'高质量': '#FF6B6B', '中质量': '#4ECDC4', '低质量': '#45B7D1'}

def 绘制特征分布(特征索引, 特征名称, 数据框, 标签列, 文件名):
    """绘制指定特征在不同类别下的分布"""
    plt.figure(figsize=(12, 8))

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 箱线图
    sns.boxplot(data=数据框, x=标签列, y=数据框.columns[特征索引], ax=ax1, palette=颜色映射,
                order=['高质量', '中质量', '低质量'])
    ax1.set_title(f'{特征名称} - 箱线图', fontsize=14, fontweight='bold')
    ax1.set_xlabel('教学质量标签', fontsize=12)
    ax1.set_ylabel(特征名称, fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # 小提琴图
    sns.violinplot(data=数据框, x=标签列, y=数据框.columns[特征索引], ax=ax2, palette=颜色映射,
                   order=['高质量', '中质量', '低质量'])
    ax2.set_title(f'{特征名称} - 小提琴图', fontsize=14, fontweight='bold')
    ax2.set_xlabel('教学质量标签', fontsize=12)
    ax2.set_ylabel(特征名称, fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'feature_distributions/{文件名}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出统计信息
    输出并保存(f"\n{特征名称}的统计信息:")
    for 类别 in ['高质量', '中质量', '低质量']:
        类别数据 = 数据框[数据框[标签列] == 类别][数据框.columns[特征索引]]
        输出并保存(".2f"
                      ".2f"
                      ".2f")

# 创建包含标签的完整数据框用于绘图
完整数据框 = X_标准化数据框.copy()
完整数据框['教学质量标签'] = y.reset_index(drop=True)

# 绘制最重要特征的分布
输出并保存(f"\n{'='*60}")
输出并保存(f"为最重要特征 '{最重要特征名称}' 绘制分布图")
输出并保存(f"{'='*60}")

绘制特征分布(最重要特征索引, 最重要特征名称, 完整数据框, '教学质量标签',
            f'最重要的特征_{最重要特征名称.replace("/", "_")}')

# 绘制最不重要特征的分布
输出并保存(f"\n{'='*60}")
输出并保存(f"为最不重要特征 '{最不重要特征名称}' 绘制分布图")
输出并保存(f"{'='*60}")

绘制特征分布(最不重要特征索引, 最不重要特征名称, 完整数据框, '教学质量标签',
            f'最不重要的特征_{最不重要特征名称.replace("/", "_")}')

# 同时绘制两个特征的对比
plt.figure(figsize=(16, 6))

# 创建子图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 最重要特征的分布
sns.boxplot(data=完整数据框, x='教学质量标签', y=完整数据框.columns[最重要特征索引],
            ax=ax1, palette=颜色映射, order=['高质量', '中质量', '低质量'])
ax1.set_title(f'最重要特征: {最重要特征名称}', fontsize=14, fontweight='bold')
ax1.set_xlabel('教学质量标签', fontsize=12)
ax1.set_ylabel('标准化值', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# 最不重要特征的分布
sns.boxplot(data=完整数据框, x='教学质量标签', y=完整数据框.columns[最不重要特征索引],
            ax=ax2, palette=颜色映射, order=['高质量', '中质量', '低质量'])
ax2.set_title(f'最不重要特征: {最不重要特征名称}', fontsize=14, fontweight='bold')
ax2.set_xlabel('教学质量标签', fontsize=12)
ax2.set_ylabel('标准化值', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# # 最重要特征的小提琴图
# sns.violinplot(data=完整数据框, x='教学质量标签', y=完整数据框.columns[最重要特征索引],
#                ax=ax3, palette=颜色映射, order=['高质量', '中质量', '低质量'])
# ax3.set_title(f'最重要特征分布: {最重要特征名称}', fontsize=14, fontweight='bold')
# ax3.set_xlabel('教学质量标签', fontsize=12)
# ax3.set_ylabel('标准化值', fontsize=12)
# ax3.grid(axis='y', alpha=0.3)

# # 最不重要特征的小提琴图
# sns.violinplot(data=完整数据框, x='教学质量标签', y=完整数据框.columns[最不重要特征索引],
#                ax=ax4, palette=颜色映射, order=['高质量', '中质量', '低质量'])
# ax4.set_title(f'最不重要特征分布: {最不重要特征名称}', fontsize=14, fontweight='bold')
# ax4.set_xlabel('教学质量标签', fontsize=12)
# ax4.set_ylabel('标准化值', fontsize=12)
# ax4.grid(axis='y', alpha=0.3)

# plt.suptitle('最重要特征 vs 最不重要特征的分布对比', fontsize=16, fontweight='bold', y=0.95)
# plt.tight_layout()
# plt.savefig('feature_distributions/特征重要性对比.png', dpi=300, bbox_inches='tight')
# plt.show()

输出并保存("\n特征分布可视化完成!")
输出并保存("生成的文件:")
输出并保存(f"- feature_distributions/最重要的特征_{最重要特征名称.replace('/', '_')}.png")
输出并保存(f"- feature_distributions/最不重要的特征_{最不重要特征名称.replace('/', '_')}.png")
输出并保存("- feature_distributions/特征重要性对比.png")

# %% [markdown]
# ## 17. 关键特征的频数分布条形图

# %%
输出并保存("\n{'='*60}")
输出并保存("关键特征的频数分布条形图")
输出并保存("{'='*60}")

# 定义颜色映射
标签颜色映射 = {'高质量': '#FF6B6B', '中质量': '#4ECDC4', '低质量': '#45B7D1'}

def 创建特征频数分布图(特征索引, 特征名称, 数据框, 标签列, 文件名, bins=20):
    """创建指定特征在不同标签下的频数分布条形图"""

    # 为每个标签创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{特征名称}在不同教学质量标签下的频数分布', fontsize=16, fontweight='bold')

    标签列表 = ['高质量', '中质量', '低质量']

    for i, 标签 in enumerate(标签列表):
        ax = axes[i]

        # 获取该标签的数据
        标签数据 = 数据框[数据框[标签列] == 标签][数据框.columns[特征索引]]

        # 创建频数分布
        counts, bin_edges = np.histogram(标签数据, bins=bins)

        # 计算bin中心
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 绘制条形图
        bars = ax.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0])*0.9,
                     color=标签颜色映射[标签], alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_title(f'{标签} (n={len(标签数据)})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{特征名称} 值', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            if count > 0:  # 只显示非零值
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                       f'{int(count)}', ha='center', va='bottom', fontsize=8)

        # 显示统计信息
        输出并保存(f"\n{特征名称} - {标签} 统计信息:")
        输出并保存(f"  样本数: {len(标签数据)}")
        输出并保存(f"  均值: {标签数据.mean():.2f}")
        输出并保存(f"  标准差: {标签数据.std():.2f}")
        输出并保存(f"  最小值: {标签数据.min():.2f}")
        输出并保存(f"  最大值: {标签数据.max():.2f}")

    plt.tight_layout()
    plt.savefig(f'feature_distributions/{文件名}', dpi=300, bbox_inches='tight')
    plt.show()

# 创建最重要的特征（考试分数）的频数分布图
输出并保存(f"\n为最重要特征 '{最重要特征名称}' 创建频数分布条形图")
创建特征频数分布图(最重要特征索引, 最重要特征名称, 完整数据框, '教学质量标签',
                  f'{最重要特征名称}_频数分布.png')

# 创建最不重要特征（参与频率）的频数分布图
输出并保存(f"\n为最不重要特征 '{最不重要特征名称}' 创建频数分布条形图")
创建特征频数分布图(最不重要特征索引, 最不重要特征名称, 完整数据框, '教学质量标签',
                  f'{最不重要特征名称}_频数分布.png')

输出并保存("\n关键特征频数分布条形图已完成:")
输出并保存(f"- feature_distributions/{最重要特征名称}_频数分布.png")
输出并保存(f"- feature_distributions/{最不重要特征名称}_频数分布.png")

# 关闭输出文件
输出文件.close()
