# -*- coding:utf-8-*-
# 导入标准库
import json
import os
# 导入 PyTorch
import torch
import sys
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
# 导入日志
from base import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class QueryClassifier:
    def __init__(self, model_path="bert_query_classifier"):
        # 初始化模型路径
        self.model_path = model_path
        # 加载 BERT 分词器
        bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained("../models/bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 记录设备信息
        logger.info(f"使用设备: {self.device}")
        # 定义标签映射
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 加载模型
        self.load_model()

    def load_model(self):
        # 检查模型路径是否存在
        if os.path.exists(self.model_path):
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 初始化新模型
            self.model = BertForSequenceClassification.from_pretrained("../models/bert-base-chinese", num_labels=2)
            # print(f'self.model--》{self.model}')
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")

    def save_model(self):
        """保存模型"""
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"模型保存至: {self.model_path}")

    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        """训练 BERT 分类模型"""
        # 加载数据集
        # print(f'os.path.exists(data_file)---》{os.path.exists(data_file)}')
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        with open(data_file, "r", encoding="utf-8") as f:
            # print(f'f.readlines()--》{f.readlines()}')
            data = [json.loads(value) for value in f.readlines()]
            # print(f'data--》{data}')
            # print(f'data--》{type(data[0])}')

        texts = [item["query"] for item in data]
        # print(f'texts--》{texts[:2]}')
        # print(f'样本个数texts--》{len(texts)}')
        labels = [item["label"] for item in data]
        # print(f'labels--》{labels[:2]}')
        # 数据划分
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        # print(f'train_texts--》{len(train_texts)}')
        # print(f'val_texts--》{len(val_texts)}')
        # 数据预处理
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        # 得到dataset对象
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)
        # print('取出一个样本', train_dataset[1])
        # print(f'验证集样本的个数--》{len(train_dataset)}')
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir="./bert_results",# 模型（检查点）以及日志保存的路径等，
            num_train_epochs=3, # 训练的轮次
            per_device_train_batch_size=8, # 训练的批次
            per_device_eval_batch_size=8, # 验证批次
            warmup_steps=20, # 学习率预热的步数
            weight_decay=0.01, # 权重衰减系数
            logging_dir="./bert_logs", # 日志保存路径:如果想生成这个文件夹，需要安装tensorboard
            logging_steps=10, # 每隔多少步打印日志
            evaluation_strategy="epoch", # 每轮都进行评估
            save_strategy="epoch", # 每轮都进行检查点的模型保存
            load_best_model_at_end=True, # 加载最优的模型
            save_total_limit=1,  # 只保存一个检查点，其他被覆盖
            metric_for_best_model="eval_loss", # 评估最优模型的指标（验证集损失）
            fp16=False,  # 禁用混合精度
        )

        # print(f'training_args--》{training_args}')
        # 初始化Trainer
        trainer = Trainer(model=self.model, args=training_args,
                          train_dataset=train_dataset, eval_dataset=val_dataset,
                          compute_metrics=self.compute_metrics)
        # 开始训练模型
        logger.info("开始训练BERT模型")
        trainer.train()
        self.save_model()

        # 对验证集进行训练好的模型验证
        # val_texts-->原始的文本；val_labels--是标签数字
        self.evaluate_model(val_texts, val_labels)


    def preprocess_data(self, texts, labels):
        """预处理数据为 BERT 输入格式"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )
        labels = [self.label_map[label] for label in labels]
        # print(f'encodings--》{encodings}')
        # print(f'encodings--》{encodings["input_ids"].shape}')
        # print(f'labels--》{labels}')
        return encodings, labels

    def create_dataset(self, encodings, labels):
        # 自定义Dataset类
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                dicts = {key: value[idx] for key, value in self.encodings.items()}
                dicts["labels"] = torch.tensor(self.labels[idx])
                return dicts

        return Dataset(encodings, labels)

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """评估模型性能"""
        # 仅对 texts 进行分词，labels 已为数字
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)
        print(f'len(dataset)-->{len(dataset)}')
        print(f'dataset[0]--->{dataset[0]}')
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        print(f'predictions--》{predictions}')
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        # print(f'pred_labels--》{type(pred_labels)}')
        # print(f'predictions.label_ids--》{predictions.label_ids}')
        true_labels = labels

        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"]
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载，记录错误
            logger.error("模型未训练或加载")
            # 默认返回通用知识
            return "通用知识"
        # 对查询进行编码
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # 将编码移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)
            # print(f'outputs--》{outputs}')
            # logits = outputs.logits
            # print(f'logits--》{logits}')
            # # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
        # 根据预测结果返回类别
        return "专业咨询" if prediction == 1 else "通用知识"
if __name__ == '__main__':
    query_classify = QueryClassifier()
    # data_file = '../classify_data/model_generic_5000.json'
    # query_classify.train_model(data_file)
    result = query_classify.predict_category(query="AI的课程大纲是什么")
    print(result)
