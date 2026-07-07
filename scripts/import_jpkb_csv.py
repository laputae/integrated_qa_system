"""Import JP学科知识问答.csv into MySQL jpkb table.

Handles multiline quoted fields that LOAD DATA INFILE cannot parse.
Usage: uv run python scripts/import_jpkb_csv.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from db_models.base import SessionLocal, engine

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mysql_qa", "data", "JP学科知识问答.csv",
)


def create_table_if_needed():
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS jpkb ("
            "  id INT AUTO_INCREMENT PRIMARY KEY,"
            "  subject_name VARCHAR(20) NOT NULL,"
            "  question VARCHAR(2000) NOT NULL,"
            "  answer TEXT NOT NULL"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        ))
        conn.commit()
    print("jpkb 表已就绪")


def import_csv():
    if not os.path.exists(CSV_PATH):
        print(f"错误：找不到 CSV 文件 {CSV_PATH}")
        sys.exit(1)

    create_table_if_needed()

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"从 CSV 读取到 {len(rows)} 条记录")

    inserted = 0
    with SessionLocal() as session:
        for i, row in enumerate(rows, start=1):
            subject = row.get("学科名称", "").strip()
            question = row.get("问题", "").strip()
            answer = row.get("答案", "").strip()

            if not question:
                print(f"  跳过第 {i} 行：问题为空")
                continue

            # Truncate subject_name to fit VARCHAR(20) if needed
            if len(subject) > 20:
                subject = subject[:20]

            try:
                session.execute(
                    text(
                        "INSERT INTO jpkb (subject_name, question, answer) "
                        "VALUES (:s, :q, :a)"
                    ),
                    {"s": subject, "q": question, "a": answer},
                )
                inserted += 1
            except Exception as e:
                print(f"  第 {i} 行插入失败: {e}")
                print(f"    subject={subject[:30]}, question={question[:50]}")

        session.commit()

    print(f"成功导入 {inserted} 条记录到 jpkb 表")


if __name__ == "__main__":
    import_csv()
