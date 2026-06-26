import os
import re
from typing import List, Optional

from langchain_text_splitters import CharacterTextSplitter
from modelscope.pipelines import pipeline

import logging

logger = logging.getLogger(__name__)


class AliTextSplitter(CharacterTextSplitter):
    """基于达摩院BERT文档语义分段模型的文本切分器。

    模型论文: https://arxiv.org/abs/2107.09278
    模型: nlp_bert_document-segmentation_chinese-base (via modelscope)
    """

    def __init__(
        self,
        pdf: bool = False,
        model_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pdf = pdf
        self._model_path = model_path or self._default_model_path()
        self._device = device
        self._pipeline = None

    @staticmethod
    def _default_model_path() -> str:
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _rag_qa_path = os.path.dirname(_current_dir)
        return os.path.join(_rag_qa_path, 'nlp_bert_document-segmentation_chinese-base')

    def _get_pipeline(self):
        if self._pipeline is None:
            logger.info(
                "Loading document-segmentation pipeline: model=%s device=%s",
                self._model_path, self._device,
            )
            self._pipeline = pipeline(
                task="document-segmentation",
                model=self._model_path,
                device=self._device,
            )
        return self._pipeline

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub(r'\s', " ", text)
            text = re.sub("\n\n", "", text)

        try:
            p = self._get_pipeline()
            result = p(documents=text)
            sent_list = [i for i in result["text"].split("\n\t") if i]
            return sent_list
        except Exception as e:
            logger.error("Semantic segmentation failed: %s", e)
            raise RuntimeError(
                f"语义切分失败 (model={self._model_path}): {e}"
            ) from e


if __name__ == '__main__':
    model_split = AliTextSplitter()
    result = model_split.split_text(text='移动端语音唤醒模型，检测关键词为"小云小云"。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用"basetrain + finetune"的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景"小云小云"数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')
    print(result)
