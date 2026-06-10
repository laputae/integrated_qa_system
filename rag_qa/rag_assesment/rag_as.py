# -*-coding:utf-8-*-
# 导入pandas库，用于数据处理和保存CSV文件
import pandas as pd
# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision, # context_relevancy(原来的写法)两者是一回事
    context_recall
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入langchain_openai的嵌入模型和聊天模型，用于评估时的语义计算和推理
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# 导入langchain_community的Ollama聊天模型和嵌入模型，用于本地模型调用
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
# 导入json库，用于加载JSON格式的评估数据集
import json, os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# 1. 加载生成的数据集
# 使用with语句打开JSON文件，确保文件正确关闭，指定编码为utf-8
with open("rag_evaluate_data.json", "r", encoding="utf-8") as f:
    # 将JSON文件内容加载到data变量中，data为包含多个数据条目的列表
    data = json.load(f)

# print(f'data--》{data}')
print(f'data--》{len(data)}')
# 2. 转换为RAGAS格式
# 创建字典eval_data，将JSON数据转换为RAGAS要求的字段格式
eval_data = {
    # 提取每个数据条目的question字段，组成问题列表
    "question": [item["question"] for item in data],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [item["context"] for item in data],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data]
}
# print(eval_data)
# 使用Dataset.from_dict将字典转换为RAGAS所需的Dataset对象
dataset = Dataset.from_dict(eval_data)
print(f'dataset--》{dataset}')

# 3. 配置RAGAS评估环境
# 初始化ChatOpenAI模型，指定使用gpt-4模型，并设置OpenAI API密钥
# llm = ChatOpenAI(model="gpt-4", openai_api_key=os.environ["OPENAI_API_KEY"])
# # 初始化OpenAI嵌入模型，用于计算语义相似度，设置API密钥
# embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# # 初始化ChatOllama模型，需要选择一个模型，设置ollama服务地址
llm = ChatOllama(model="qwen2.5:7b", base_url='http://localhost:11434')
# embeddings = OllamaEmbeddings(model="qwen2.5:7b",base_url='http://localhost:11434' )
embeddings = OllamaEmbeddings(model="mxbai-embed-large",base_url='http://localhost:11434')


# 4. 执行评估
# 调用evaluate函数，传入数据集、评估指标、LLM模型和嵌入模型
result = evaluate(
    # 传入转换好的Dataset对象
    dataset=dataset,
    # 指定使用的评估指标列表
    metrics=[
        faithfulness,  # 忠实度：答案是否基于上下文
        answer_relevancy,  # 答案相关性：答案与问题的匹配度
        context_precision,  # 上下文相关性：上下文是否仅包含相关信息
        context_recall  # 上下文召回率：上下文是否包含所有必要信息
    ],
    # 传入配置好的LLM模型
    llm=llm,
    # 传入配置好的嵌入模型
    embeddings=embeddings
)

# 5. 输出并保存结果
# 打印评估结果标题
print("RAGAS评估结果：")
# 打印评估结果，包含各指标的分数
print(result)







