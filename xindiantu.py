import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ====================== 全局配置（修复中文字体缺失） ======================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

# 统一四类配色
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")

# 1. 数据源：桌面 train.csv
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
csv_path = os.path.join(desktop, "train.csv")

# 2. 输出文件夹：桌面 test1 （所有图片、模型、日志、提交文件全部存这里）
output_dir = os.path.join(desktop, "test1")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出文件夹：桌面/test1")

# 所有输出文件路径
submit_csv = os.path.join(output_dir, "sample_submit.csv")
best_model_path = os.path.join(output_dir, "best_ecg_model.pth")
train_log_path = os.path.join(output_dir, "train_log.csv")
wave_stat_csv = os.path.join(output_dir, "wave_statistics.csv")

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ====================== 1. 读取数据 + 原始四类分布可视化 ======================
df = pd.read_csv(csv_path)
print("===== 数据集基础信息 =====")
print(f"总样本量：{df.shape[0]}")
print("数据表列名：", df.columns.tolist())
# 仅4类标签统计
origin_label_cnt = df["label"].value_counts().sort_index()
print("\n===== 原始4类标签样本数量分布 =====")
print(origin_label_cnt)

# 绘制原始四类样本柱状图
plt.figure(figsize=(8, 3))
plt.bar(origin_label_cnt.index.astype(str), origin_label_cnt.values, color=color_list)
plt.title("心电数据集 原始4类标签样本数量分布")
plt.xlabel("类别 label(0/1/2/3)")
plt.ylabel("样本数量")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "origin_4class_dist.png"))
plt.show()


# ====================== 2. 解析逗号分隔心电波形 ======================
def parse_ecg_wave(s):
    """解析逗号分隔的时序心电信号"""
    if pd.isna(s) or str(s).strip() == "":
        return None
    try:
        str_list = str(s).strip().split(",")
        wave_arr = np.array([float(x) for x in str_list if x.strip()], dtype=np.float64)
        return wave_arr
    except Exception as e:
        return None


# 解析波形
df["signal_arr"] = df["heartbeat_signals"].apply(parse_ecg_wave)
# 过滤解析失败的空数据
df_valid = df[df["signal_arr"].notna()].reset_index(drop=True)
print(f"\n波形解析完成，有效样本数：{len(df_valid)}")

# 统一时序长度（取中位数长度截断/补零，保证输入维度一致）
wave_lengths = [len(arr) for arr in df_valid["signal_arr"]]
fix_len = int(np.median(wave_lengths))
print(f"统一波形输入长度：{fix_len}")


def fix_wave_length(arr, target_len):
    if len(arr) >= target_len:
        return arr[:target_len]
    else:
        pad = np.zeros(target_len - len(arr))
        return np.concatenate([arr, pad])


# 标准化 + 统一长度
signal_list = []
label_list = []
for idx, row in df_valid.iterrows():
    wave = fix_wave_length(row["signal_arr"], fix_len)
    # 标准化
    mean = np.mean(wave)
    std = np.std(wave)
    if std < 1e-6:
        norm_wave = wave - mean
    else:
        norm_wave = (wave - mean) / std
    signal_list.append(norm_wave)
    label_list.append(row["label"])

ecg_raw = np.array(signal_list)
labels = np.array(label_list)
NUM_CLASSES = 4

all_df = pd.DataFrame({"signal": signal_list, "label": labels})
cls_dist = all_df["label"].value_counts().sort_index()
print("\n===== 清洗后4类样本分布 =====")
print(cls_dist)

# ====================== 新增可视化1：各类波形长度分布直方图 ======================
plt.figure(figsize=(10, 4))
wave_len_by_cls = {}
for c in range(NUM_CLASSES):
    cls_data = all_df[all_df["label"] == c]
    lengths = [len(w) for w in cls_data["signal"]]
    wave_len_by_cls[c] = lengths
    plt.hist(lengths, alpha=0.5, label=f"类别{c}", color=color_list[c], bins=20)
plt.xlabel("单条波形采样点总数")
plt.ylabel("样本数量")
plt.title("四类心电波形采样长度分布直方图")
plt.legend()
plt.grid(alpha=0.3)
len_hist_path = os.path.join(output_dir, "wave_length_dist.png")
plt.savefig(len_hist_path, dpi=150)
plt.show()
print(f"波形长度分布图已保存：{len_hist_path}")

# ====================== 新增可视化2：四类心电典型波形对比图 ======================
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
for label_c in range(NUM_CLASSES):
    # 筛选当前类所有样本，随机取1条波形
    cls_data = all_df[all_df["label"] == label_c]
    rand_sample = cls_data.sample(n=1, random_state=SEED)
    wave_data = rand_sample["signal"].iloc[0]
    x_axis = list(range(len(wave_data)))

    axes[label_c].plot(x_axis, wave_data, color=color_list[label_c], linewidth=1.2)
    axes[label_c].set_title(f"类别 {label_c} 典型心电波形", fontsize=12)
    axes[label_c].set_xlabel("采样点序号")
    axes[label_c].set_ylabel("归一化幅值")
    axes[label_c].grid(alpha=0.3)

plt.suptitle("四类心电信号典型波形对比图", fontsize=16, y=0.98)
plt.tight_layout()
wave_save_path = os.path.join(output_dir, "ecg_4wave_compare.png")
plt.savefig(wave_save_path, dpi=150)
plt.show()
print(f"四类波形对比图已保存：{wave_save_path}")

# ====================== 新增量化统计3：波形均值、标准差、最大幅值 ======================
stat_result = {}
print("\n===== 四类心电波形量化统计指标（均值/标准差/最大峰值） =====")
for c in range(NUM_CLASSES):
    cls_waves = all_df[all_df["label"] == c]["signal"].tolist()
    all_vals = np.concatenate(cls_waves)
    mean_val = np.round(np.mean(all_vals), 4)
    std_val = np.round(np.std(all_vals), 4)
    peak_max = np.round(np.max(np.abs(all_vals)), 4)
    stat_result[c] = {"均值": mean_val, "标准差": std_val, "最大绝对幅值": peak_max}
    print(f"类别{c}：均值={mean_val}，标准差={std_val}，最大幅值={peak_max}")
stat_df = pd.DataFrame(stat_result).T
stat_df.to_csv(wave_stat_csv)
print(f"四类波形统计指标已保存至：{wave_stat_csv}")

# 计算类别权重，缓解不平衡
total = len(all_df)
cls_weight = []
for c in range(NUM_CLASSES):
    cnt = cls_dist[c]
    w = total / (NUM_CLASSES * cnt)
    cls_weight.append(w)
cls_weight_tensor = torch.tensor(cls_weight, dtype=torch.float32).to(device)
print(f"类别平衡权重[0,1,2,3]：{np.round(cls_weight, 4)}")

# ====================== 3. 分层8:2划分训练集/测试集 ======================
train_df, test_df = train_test_split(
    all_df, test_size=0.2, random_state=SEED, stratify=all_df["label"]
)
print(f"\n===== 数据集划分 =====")
print(f"训练集：{len(train_df)} 条 | 测试集：{len(test_df)} 条")
print("训练集类别分布：")
print(train_df["label"].value_counts().sort_index())

# 提取训练数据用于SMOTE均衡
train_X_raw = np.vstack(train_df["signal"])
train_y_raw = train_df["label"].values
# SMOTE插值平衡训练集
smote = SMOTE(random_state=SEED, k_neighbors=5)
train_X_bal, train_y_bal = smote.fit_resample(train_X_raw, train_y_raw)
bal_dist = pd.Series(train_y_bal).value_counts().sort_index()
print("\nSMOTE均衡后训练集各类样本数量：")
print(bal_dist)


# 训练集简单数据增强（波形小幅缩放+偏移）
def wave_augment(wave):
    scale = np.random.uniform(0.8, 1.2)
    offset = np.random.uniform(-0.1, 0.1)
    return wave * scale + offset


train_aug = np.array([wave_augment(s) for s in train_X_bal])
train_signals = train_aug
train_labels = train_y_bal

# 测试集不增强、不做SMOTE
test_signals = np.vstack(test_df["signal"])
test_labels = test_df["label"].values

# 转换模型输入维度 [N, Channel, Length]
X_train = torch.tensor(train_signals, dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
X_test = torch.tensor(test_signals, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(test_labels, dtype=torch.long).to(device)

BATCH_SIZE = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# ====================== 4. 多层1D残差卷积网络（适配心电时序） ======================
class ECG1DCNN(nn.Module):
    def __init__(self, seq_len, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


model = ECG1DCNN(fix_len, NUM_CLASSES).to(device)
# 加权交叉熵损失，解决类别不平衡
criterion = nn.CrossEntropyLoss(weight=cls_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
MAX_EPOCH = 50
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=1e-5)
softmax = nn.Softmax(dim=1)

# 早停参数
PATIENCE = 8
best_acc = 0.0
stop_cnt = 0
train_records = []
loss_history = []
acc_history = []

print("\n===== 有监督心电分类训练开始（4类标签，逗号分隔波形） =====")
for epoch in range(MAX_EPOCH):
    # 训练阶段
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.shape[0]
    avg_loss = total_loss / len(train_loader.dataset)
    loss_history.append(avg_loss)

    # 测试集评测
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())
    test_acc = accuracy_score(all_true, all_pred)
    recall_arr = recall_score(all_true, all_pred, average=None)
    acc_history.append(test_acc)

    # 学习率更新
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    train_records.append({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "test_acc": test_acc,
        "lr": current_lr
    })
    print(f"Epoch {epoch + 1:2d} | Loss:{avg_loss:.4f} | TestAcc:{test_acc:.4f} | LR:{current_lr:.6f}")
    print(f"每类召回率[0,1,2,3]：{np.round(recall_arr, 4)}")

    # 保存最优模型 + 早停判断
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"更新最优模型，最高测试准确率：{best_acc:.4f}")
        stop_cnt = 0
    else:
        stop_cnt += 1
        if stop_cnt >= PATIENCE:
            print(f"早停触发，连续{PATIENCE}轮准确率无提升，训练终止")
            break

# 保存训练日志
log_df = pd.DataFrame(train_records)
log_df.to_csv(train_log_path, index=False)

# ====================== 绘制训练曲线 ======================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss_history) + 1), loss_history, c="#d62728", label="训练损失")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练损失变化曲线")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(acc_history) + 1), acc_history, c="#2ca02c", label="测试集准确率")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("测试集准确率变化曲线")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "train_curve.png"))
plt.show()

# ====================== 绘制测试集混淆矩阵 ======================
cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1", "2", "3"], yticklabels=["0", "1", "2", "3"])
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("测试集4分类混淆矩阵")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()

# ====================== 输出竞赛提交CSV（测试集预测概率） ======================
model.eval()
with torch.no_grad():
    test_out = model(X_test)
    prob_matrix = softmax(test_out).cpu().numpy()

submit_df = pd.DataFrame({
    "id": test_df.index.values,
    "label_0": prob_matrix[:, 0],
    "label_1": prob_matrix[:, 1],
    "label_2": prob_matrix[:, 2],
    "label_3": prob_matrix[:, 3]
})
submit_df.to_csv(submit_csv, index=False, float_format="%.6f")

# ====================== 运行结束提示 ======================
print("\n" + "=" * 70)
print(f"所有输出文件统一存储位置：桌面/test1 文件夹")
print("文件夹内全部输出文件清单：")
print("1. origin_4class_dist.png   原始4类样本数量柱状图（原始数据分析图）")
print("2. wave_length_dist.png     四类波形采样长度分布直方图")
print("3. ecg_4wave_compare.png    四类心电典型波形对比图")
print("4. wave_statistics.csv      四类波形均值、标准差、峰值量化统计表")
print("5. train_curve.png          训练损失+测试准确率变化曲线")
print("6. confusion_matrix.png     测试集混淆矩阵热力图")
print("7. sample_submit.csv        测试集预测概率，竞赛提交文件")
print("8. best_ecg_model.pth       训练最优模型权重文件")
print("9. train_log.csv            每轮epoch损失、准确率、学习率日志")
print(f"测试集最优准确率：{best_acc:.4f}")
print("=" * 70)
print("测试集前5行预测概率预览：")
print(submit_df.head())