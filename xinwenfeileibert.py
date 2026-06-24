import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

# ===================== 离线环境配置 =====================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ===================== 全局路径配置 =====================
DESKTOP = r"C:\Users\asus\Desktop"
OUTPUT_DIR = os.path.join(DESKTOP, "test2")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
RAW_DATA = os.path.join(DESKTOP, "train_set.csv")
BERT_LOCAL_PATH = r"C:\Users\asus\Desktop\bert-base-chinese"

# ===================== 文本清洗函数（保留标点，不破坏语义） =====================
def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", "", text)  # 去除HTML标签
    text = re.sub(r"http\S+|www.\S+", "", text)  # 去除链接
    text = re.sub(r"@\w+|#.+?#", "", text)  # 去除@用户、#话题#
    text = re.sub(r"\s+", " ", text)  # 合并多余空格
    return text.strip()

# ===================== 导入依赖 =====================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler

# ===================== 预编码专用数据集类（张量切片，消除内存碎片） =====================
class NewsDataset(Dataset):
    def __init__(self, input_tensor, mask_tensor, label_list):
        self.input_tensor = input_tensor
        self.mask_tensor = mask_tensor
        self.labels = label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_tensor[idx],
            "attention_mask": self.mask_tensor[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ===================== 全局预编码工具函数 =====================
def pre_encode_all_texts(text_list, tokenizer, max_len):
    input_ids_cache = []
    mask_cache = []
    for text in tqdm(text_list, desc="全局预编码文本，仅执行一次"):
        res = tokenizer(
            text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids_cache.append(res["input_ids"].flatten())
        mask_cache.append(res["attention_mask"].flatten())
    # 转为连续张量，大幅降低加载开销
    input_tensor = torch.stack(input_ids_cache)
    mask_tensor = torch.stack(mask_cache)
    return input_tensor, mask_tensor

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 提速+高精度平衡超参
    MAX_LEN = 192
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()
    best_test_acc = 0
    model_save_path = os.path.join(OUTPUT_DIR, "best_bert_model.pt")

    print(f"\n正在使用设备：{DEVICE}")
    if torch.cuda.is_available():
        print(f"显卡：{torch.cuda.get_device_name(0)}")

    # ========== 1. 数据集数据分析 & 绘图 ==========
    print("=" * 60)
    print("数据集数据分析（Numpy数值统计）")
    print("=" * 60)
    df_raw = pd.read_csv(RAW_DATA, sep="\t")
    print(f"原始数据读取后总量：{df_raw.shape[0]}行")

    # 【核心修复1：原始数据集全局去重，彻底消除重复文本】
    df_raw = df_raw.drop_duplicates(subset=["text"], keep="first")
    print(f"原始文本去重完成，剩余总量：{df_raw.shape[0]}行")

    print("缺失值统计：\n", df_raw.isnull().sum())

    df_analyze = df_raw[["label", "text"]].copy()
    text_len_list = [len(str(t)) for t in df_analyze["text"]]
    text_len_np = np.array(text_len_list)

    print("\n【Numpy文本长度指标】")
    print(f"平均长度：{np.mean(text_len_np):.2f}")
    print(f"最短文本：{np.min(text_len_np)}")
    print(f"最长文本：{np.max(text_len_np)}")
    print(f"中位数：{np.median(text_len_np):.2f}")
    print(f"标准差：{np.std(text_len_np):.2f}")

    label_counts = df_analyze["label"].value_counts().sort_index()
    print("\n各类别样本数量：\n", label_counts)
    print(f"分类总数：{df_analyze['label'].nunique()}")

    # 类别分布图
    plt.figure(figsize=(10, 5))
    plt.bar(np.array(label_counts.index, dtype=str), label_counts.values, color="#4472C4")
    plt.title("各类别样本数量分布")
    plt.xlabel("标签类别")
    plt.ylabel("样本数")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "label_dist.png"), dpi=200)
    plt.close()

    # 文本长度分布图
    plt.figure(figsize=(10, 5))
    plt.hist(text_len_np, bins=30, color="#ED7D31", alpha=0.7)
    plt.title("文本长度分布直方图")
    plt.xlabel("字符长度")
    plt.ylabel("样本数量")
    plt.grid(axis="y", alpha=0.3)
    cut_max = np.percentile(text_len_np, 99)
    plt.xlim(0, cut_max)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "text_len_dist.png"), dpi=200)
    plt.close()

    print(f"\n数据集分析独立图片已保存至：{OUTPUT_DIR}")
    print("=" * 60)
    print("数据分析完成，进入文本预处理")
    print("=" * 60)

    # ========== 2. 数据清洗与8:2分层划分 ==========
    df = df_raw[["label", "text"]].dropna()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]

    # 【核心修复2：更换随机种子，避免固定划分重复抽取重叠样本】
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=123, stratify=df["label"]
    )
    NUM_LABELS = df["label"].nunique()
    print(f"清洗后总样本：{len(df)}")
    print(f"训练集(80%)：{len(train_df)}，测试集(20%)：{len(test_df)}")
    print("自动识别分类数量：", NUM_LABELS)

    # ===================== 数据集泄露校验代码 =====================
    print("\n===== 【数据集泄露校验开始】 =====")
    train_text_list = train_df["text"].astype(str).tolist()
    test_text_list = test_df["text"].astype(str).tolist()
    train_text_set = set(train_text_list)
    test_text_set = set(test_text_list)
    repeat_text = train_text_set & test_text_set

    print(f"训练集总样本：{len(train_text_list)}")
    print(f"测试集总样本：{len(test_text_list)}")
    print(f"训练/测试重复文本数量：{len(repeat_text)}")
    if len(repeat_text) > 0:
        print(" 严重警告：存在样本泄露！测试集文本出现在训练集，测试准确率会虚高")
        print("重复样例示例：", list(repeat_text)[:3])
    else:
        print(" 训练、测试文本完全隔离，无样本泄露")

    print("\n测试集标签分布：")
    test_label_dist = test_df["label"].value_counts().sort_index()
    print(test_label_dist)
    if len(test_label_dist) == 1:
        print(" 警告：测试集仅有单一标签，分类测试无意义，数据读取异常！")

    print("\n随机抽取训练集3条样本：")
    print(train_df[["text", "label"]].sample(3))
    print("\n随机抽取测试集3条样本：")
    print(test_df[["text", "label"]].sample(3))
    print("===== 【数据集泄露校验结束】 =====")
    # ==================================================================================

    # 关闭类别加权，避免精度偏移
    loss_fn = nn.CrossEntropyLoss()

    # ========== 3. 加载本地BERT & 预编码全部文本 ==========
    print("\n 正在从本地加载 BERT 模型...")
    tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        BERT_LOCAL_PATH,
        num_labels=NUM_LABELS,
        local_files_only=True,
        hidden_dropout_prob=0.3
    ).to(DEVICE)

    # 全局预编码并转为连续张量（核心提速改动）
    print("\n开始预编码训练集文本...")
    train_input_tensor, train_mask_tensor = pre_encode_all_texts(train_df["text"].tolist(), tokenizer, MAX_LEN)
    print("开始预编码测试集文本...")
    test_input_tensor, test_mask_tensor = pre_encode_all_texts(test_df["text"].tolist(), tokenizer, MAX_LEN)

    # 构造数据集
    train_dataset = NewsDataset(train_input_tensor, train_mask_tensor, train_df["label"].tolist())
    test_dataset = NewsDataset(test_input_tensor, test_mask_tensor, test_df["label"].tolist())

    # ========== 4. DataLoader 安全多进程，无卡死风险 ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # ========== 5. 分层学习率（主干与分类头差距缩小，保证高精度） ==========
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": 2e-4,
            "lr": LR
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": 0.0,
            "lr": LR
        },
        {"params": list(model.classifier.parameters()), "lr": LR * 2}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.05), num_training_steps=total_steps)

    # ========== 6. 训练循环（全部高精度约束保留，新增最优模型保存） ==========
    print("\n 开始 BERT 训练...")
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0
        pred_tensor_list = []
        true_tensor_list = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} 训练"):
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            with autocast():
                outputs = model(input_ids, attention_mask=mask)
                loss = loss_fn(outputs.logits, label)
            total_train_loss += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_pred = torch.argmax(outputs.logits, dim=1)
            pred_tensor_list.append(batch_pred)
            true_tensor_list.append(label)
        # 统一计算指标
        train_preds = torch.cat(pred_tensor_list).cpu().numpy()
        train_trues = torch.cat(true_tensor_list).cpu().numpy()
        train_acc = accuracy_score(train_trues, train_preds)
        avg_train_loss = total_train_loss / len(train_loader)

        # 测试阶段
        model.eval()
        total_test_loss = 0
        pred_tensor_list = []
        true_tensor_list = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} 测试"):
                input_ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                with autocast():
                    outputs = model(input_ids, attention_mask=mask)
                    loss = loss_fn(outputs.logits, label)
                total_test_loss += loss.item()
                batch_pred = torch.argmax(outputs.logits, dim=1)
                pred_tensor_list.append(batch_pred)
                true_tensor_list.append(label)
        test_preds = torch.cat(pred_tensor_list).cpu().numpy()
        test_trues = torch.cat(true_tensor_list).cpu().numpy()
        test_acc = accuracy_score(test_trues, test_preds)
        avg_test_loss = total_test_loss / len(test_loader)

        # 保存最优模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"\n 保存当前最优模型，最佳测试准确率：{best_test_acc:.4f}")

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(avg_train_loss)
        test_loss_list.append(avg_test_loss)

        print(f"\n Epoch {epoch+1}")
        print(f"训练 Loss：{avg_train_loss:.4f} | 训练 Acc：{train_acc:.4f}")
        print(f"测试 Loss：{avg_test_loss:.4f} | 测试 Acc：{test_acc:.4f}")

    # ========== 7. 保存4张独立曲线图 ==========
    # 训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(train_loss_list), label="训练损失", color="r")
    plt.title("训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "train_loss.png"), dpi=200)
    plt.close()

    # 测试损失
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(test_loss_list), label="测试损失", color="orange")
    plt.title("测试损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_loss.png"), dpi=200)
    plt.close()

    # 训练准确率
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(train_acc_list), label="训练准确率", color="b")
    plt.title("训练准确率曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "train_acc.png"), dpi=200)
    plt.close()

    # 测试准确率
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(test_acc_list), label="测试准确率", color="g")
    plt.title("测试准确率曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_acc.png"), dpi=200)
    plt.close()

    print(f"\n 4张训练指标独立图片已保存至：{OUTPUT_DIR}")

    # ========== 8. 导出完整预测结果CSV ==========
    model.eval()
    test_pred = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                outputs = model(input_ids, attention_mask=mask)
            batch_pred = torch.argmax(outputs.logits, dim=1)
            test_pred.append(batch_pred)
    test_pred = torch.cat(test_pred).cpu().numpy()

    # 完整详情结果（原有文件）
    test_result = test_df.copy()
    test_result["pred_label"] = test_pred
    test_result.to_csv(os.path.join(OUTPUT_DIR, "bert_test_result.csv"), index=False, encoding="utf-8-sig")

    # ========== 新增：生成提交文件 sample_submit.csv 保存到桌面test2 ==========
    # 仅保留文本、预测标签，简洁提交格式
    submit_df = pd.DataFrame()
    submit_df["text"] = test_df["text"].tolist()
    submit_df["pred_label"] = test_pred
    # 保存路径：桌面/test2/sample_submit.csv
    submit_path = os.path.join(OUTPUT_DIR, "sample_submit.csv")
    submit_df.to_csv(submit_path, index=False, encoding="utf-8-sig")
    print(f"\n 提交文件 sample_submit.csv 已生成，路径：{submit_path}")

    print("\n BERT 训练全部完成！所有图片、结果文件均保存在桌面test2文件夹！")