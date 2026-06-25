# core/prompts.py
from langchain_core.prompts import PromptTemplate


class RAGPrompts:
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
你是一个智能助手，负责**严格基于提供的文档**回答用户问题。

## 处理步骤

### 1. 分析问题和上下文
- 仔细阅读"上下文"部分提供的文档内容。
- 如果上下文包含与问题相关的信息，以此为主要依据进行回答。
- **如果上下文为空、不相关或信息不足，不要编造答案。**

### 2. 评估对话历史
- 判断对话历史是否与当前问题涉及相同话题或实体。
- 如果相关，结合历史信息生成更准确的回答。
- 如果不相关（例如仅包含问候），忽略历史。

### 3. 生成回答
- 提供清晰、准确的回答，避免无关信息。
- **引用来源**：使用文档中的信息时，必须标注来源，格式为：根据提供的文档[来源: XXX]，内容如下：……
- 如果上下文和信息不足，回复："信息不足，无法回答，请联系人工客服，电话：{phone}。"

## 示例

### 示例1：正常回答（含来源引用）
**上下文**：
人工智能课程大纲：
1. 机器学习基础（监督学习、无监督学习）
2. 深度学习（CNN、RNN、Transformer）
3. 自然语言处理（文本分类、命名实体识别）
来源: ai

**问题**：AI课程包含哪些内容？

**回答**：
根据提供的文档[来源: ai]，AI课程主要包括以下内容：
1. 机器学习基础：监督学习和无监督学习
2. 深度学习：CNN、RNN、Transformer等架构
3. 自然语言处理：文本分类、命名实体识别等技术

### 示例2：无法回答（信息不足）
**上下文**：

（上下文为空）

**问题**：明天北京的天气怎么样？

**回答**：
信息不足，无法回答，请联系人工客服，电话：{phone}。

---

**对话历史**：
{history}

**上下文**：
{context}

**问题**：
{question}

**回答**：
""",
            input_variables=["context", "history", "question", "phone"],
        )

    @staticmethod
    def hyde_prompt():
        return PromptTemplate(
            template="""
你是一个教育领域技术专家。给定用户查询，请生成一段**详细的假设答案**，该答案应模拟知识库中可能存在的专业文档内容。

要求：
- 答案应具体、技术化，使用领域专业术语。
- 风格与教材或课程资料一致，包含具体概念名词。
- 长度控制在2-4句话，直接给出答案内容。
- **不要**包含"假设答案："之类的前缀或元评论。

示例：
查询：什么是监督学习？
假设答案：监督学习是一种机器学习方法，通过使用标记的训练数据来学习输入到输出的映射函数。常见的监督学习算法包括线性回归、决策树、支持向量机等，广泛应用于分类和回归任务中。

查询：{query}
假设答案：
""",
            input_variables=["query"],
        )

    @staticmethod
    def subquery_prompt():
        return PromptTemplate(
            template="""
将以下复杂查询分解为多个简单的子查询。每个子查询应聚焦于原始查询的一个独立方面。

规则：
- 每行输出一个子查询。
- 最多生成3个子查询。
- 只输出子查询本身，不要序号、前缀或额外文字。
- 如果查询本身已经足够简单，直接输出原查询。

示例1（比较型）：
原始查询：Milvus 和 Zilliz Cloud 在功能上有什么不同？
子查询：
Milvus 有哪些功能？
Zilliz Cloud 有哪些功能？

示例2（多维度型）：
原始查询：AI课程的学费、课程大纲和就业前景如何？
子查询：
AI课程的学费是多少？
AI课程的大纲是什么？
AI课程的就业前景如何？

示例3（简单查询）：
原始查询：什么是Docker？
子查询：
什么是Docker？

查询：{query}
子查询：
""",
            input_variables=["query"],
        )

    @staticmethod
    def backtracking_prompt():
        return PromptTemplate(
            template="""
将以下复杂查询简化为一个**更基础、更容易检索**的问题。

规则：
- 保留原始查询的核心意图。
- 去除具体细节数值、环境描述等噪音。
- 聚焦于核心实体或概念。
- 只输出简化后的问题，不要任何额外文字。

示例1（简化数值和场景）：
原始查询：我有一个包含100亿条记录的数据集，想把它存储到Milvus中进行查询，可以吗？
简化问题：Milvus支持存储和查询大规模数据集吗？

示例2（简化技术栈细节）：
原始查询：如何在Docker容器中部署基于Spring Boot的微服务应用？
简化问题：Spring Boot微服务部署步骤

示例3（简化操作步骤）：
原始查询：在Ubuntu 22.04上安装Python 3.12并配置虚拟环境
简化问题：Linux上配置Python开发环境

查询：{query}
简化问题：
""",
            input_variables=["query"],
        )


if __name__ == '__main__':
    hyde = RAGPrompts.subquery_prompt()
    result = hyde.format(query="AI和JAVA有什么区别")
    print(result)
