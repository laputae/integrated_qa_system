# import os
# import os
# # from base import Config, logger
# # conf = Config()
# # print(conf.REDIS_PORT)
# # logger.info('这是岑氏')
# # print("INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)" % (1, 2, 3))
# #
# # print("它的名字是%s" % ("张三"))
#
# # class Fun():
# #     def _getdata(self):
# #         return '你好'
# #
# # fun = Fun()
# # print(fun._getdata())
# #
# # a = None
# # if not a:
# #     print('你好')
# # query = "你好"
# # query = 2
# # if isinstance(query, int):
# #     print('aaa')
#
# # if query is str:
# #     print('ndada')
#
# # import numpy as np
# #
# # scores = [1.2, 2.4, 3.6]
# # print(np.max(scores))
# # result1 = np.exp(scores-np.max(scores))
# # print(result1)
# #
# # print(result1/result1.sum())
#
# # # 遍历指定目录及其子目录
# # directory_path = "/Users/ligang/Desktop/EduRAG课堂资料/codes/integrated_qa_system/rag_qa/data/ai_data"
# # for root, _, files in os.walk(directory_path):
# #     print(f'root---》{root}')
# #     print(f'_---》{_}')
# #     print(f'files---》{files}')
# #     print('*'*80)
#
# # from datetime import datetime
# # print(datetime.now())
# # a = datetime.now().isoformat()
# # b = datetime.fromisoformat(a)
# # print(b)
#
# # a = []
# # b = [2, 3]
# # a.extend(b)
# # print(a)
#
# # a = (1.2, 4.6, 4.9)
# # b = ('def', 'gdef', 'ef')
# # print(sorted(zip(a, b), reverse=True))
#
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained('/Users/ligang/PycharmProjects/LLM/Itcast_qa_system/models/bge-reranker-large')
# model = AutoModelForSequenceClassification.from_pretrained('/Users/ligang/PycharmProjects/LLM/Itcast_qa_system/models/bge-reranker-large')
# model.eval()
#
# pairs = [['what is panda?', "what is panda?"],
#          ['what is panda?', 'hi'],
#          ['what is panda?', 'The giant panda '],
#          ]
# with torch.no_grad():
#     inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
#     scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
#     print(scores)
#
# from sentence_transformers import CrossEncoder
# model1 = CrossEncoder(model_name="/Users/ligang/PycharmProjects/LLM/Itcast_qa_system/models/bge-reranker-large")
# print(model1.predict(pairs))
#
# # pairs = [
# #     ['what is panda?', "what is panda?"],
# #     ['what is panda?', 'hi'],
# #     ['what is panda?', 'The giant panda ']
# # ]
# # #
# # with torch.no_grad():
# #     inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
# #     raw_logits = model(**inputs).logits.view(-1)  # 原始 logits
# #
# #     # 模拟 CrossEncoder 的后处理（BGE-Reranker 通常是 sigmoid + 缩放）
# #     normalized_scores = torch.sigmoid(raw_logits)  # 映射到 [0, 1]
# #     scaled_scores = normalized_scores * 10         # 缩放到 [0, 10]（常见做法）
# #
# # print("Raw logits:", raw_logits.tolist())
# # print("Sigmoid + Scale:", scaled_scores.tolist())
# from milvus_model.hybrid import BGEM3EmbeddingFunction
# embedding_function = BGEM3EmbeddingFunction(model_name_or_path="./rag_qa/bge-m3", use_fp16=False, device="cpu")
# print(f"self.embedding_function.dim--》{embedding_function.dim}")


# list1 = [0.6, 0.8, 0.7]
# list2 = ['dad', 'tad', 'kda']
# # print(list(zip(list1, list2)))
#
# a = sorted(zip(list1, list2), reverse=True)
# print(a)
# import numpy as np
# logits = np.array([[0.6, 0.4],
#                    [0.3, 0.7],
#                    [0.8, 0.2]])
# labels = np.array([1, 1, 0])
#
# predictions = np.argmax(logits, axis=-1)
# print(f'predictions--->{predictions}')
# print((predictions==labels).mean())

'''
如果是自己自定义的模型
保存model
torch.save(model.state_dict(), 'path')
加载模型
model.load_state_dict(torch.load('pth'))


如果只使用预训练模型的框架（完全没有自定义）：
保存模型
model.save_pretrained(self.model_path)
# 加载模型
BertModel.from_pretrained(self.model_path)

tokenizer.save_pretrained(self.model_path)
BertTokenizer.from_pretrained(self.model_path)
'''


# def fun():
#     for i in range(3):
#         yield i
#
#
# def fun1():
#     a = ''
#     for value in fun():
#         a += str(value)
#         yield value
#     print('a', a)
#
# for value in fun1():
#     print(value)

# subqueries_text = "\nAI是什么 \nJAVA是什么 \n"
# print(subqueries_text)
# a = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
# print(a)


# 基于内容去重
# all_docs = [{"page_content": "今天天气很好", "source": "doc1"},
#             {"page_content": "今天天气很好", "source": "doc2"},
#             {"page_content": "我喜欢编程", "source": "doc3"},
#             {"page_content": "我喜欢编程", "source": "doc4"},
#             {"page_content": "python很有趣", "source": "doc5"},]
#
# unique_docs_list = {doc["page_content"]: doc for doc in all_docs}
# print(unique_docs_list)
# print(list(unique_docs_list.values()))

# a = 4
# c = a//2
# print(c)
# print(type(c))


# import argparse
#
# # 1. 创建解析器
# parser = argparse.ArgumentParser(description='简要描述这个脚本的功能')
#
# # 2. 添加参数
# parser.add_argument('input', help='输入文件路径')  # 位置参数
# parser.add_argument('--output', help='输出文件路径')  # 可选参数
# parser.add_argument('--verbose', action='store_true', help='是否输出详细信息')
#
# # 3. 解析参数
# args = parser.parse_args()
# print(f'args--》{args}')
# # 4. 使用参数
# print(f"输入文件: {args.input}")
# if args.output:
#     print(f"输出文件: {args.output}")
# if args.verbose:
#     print("详细模式已开启")

def fun():
    for i in range(3):
        yield i+int(4)

def fun1():
    for j in fun():
        yield j

# alist = [{"time1":'5'}, {"time2":"4"}]
# print(alist[::-1])
for value in fun1():
    print(value)