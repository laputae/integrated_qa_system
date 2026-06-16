# -*- coding:utf-8 -*-
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)

from sqlalchemy import text
from sqlalchemy.orm import Session

from base import logger


class MySQLClient:
    def __init__(self, engine=None):
        self.logger = logger
        if engine is not None:
            self.engine = engine
        else:
            from db_models.base import engine as default_engine
            self.engine = default_engine

    def fetch_questions(self):
        try:
            with Session(self.engine) as session:
                result = session.execute(text("SELECT question FROM jpkb"))
                return [(row[0],) for row in result.fetchall()]
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            return []

    def fetch_answer(self, question):
        if isinstance(question, tuple):
            question = question[0]
        try:
            with Session(self.engine) as session:
                result = session.execute(
                    text("SELECT answer FROM jpkb WHERE question = :q"),
                    {"q": question},
                )
                row = result.fetchone()
                return row[0] if row else None
        except Exception as e:
            self.logger.error(f"答案获取失败: {e}")
            return None
