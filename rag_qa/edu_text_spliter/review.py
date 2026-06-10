# class A():
#     def a1(self):
#         print('你好A')
#     def b1(self):
#         self.a1()
#
# class B(A):
#     def a1(self,):
#         print('你好B')
#
# b = B()
# b.b1()


from transformers import BertModel
bert_model = BertModel.from_pretrained('/Users/ligang/Desktop/EduRAG课堂资料/codes/integrated_qa_system/rag_qa/nlp_bert_document-segmentation_chinese-base')
print(bert_model)

