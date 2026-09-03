#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通过 Linux 命令行参数将病人及药品信息导入 MySQL。

运行示例：

python3 import_medicine.py \
    --patient "张三" \
    --medicines "阿莫西林,布洛芬,维生素C"
"""

import argparse
import json
import re
import sys

import pymysql


# ==================== Editable config ====================
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = "root"
DB_NAME = "medicine_db"
DB_CHARSET = "utf8"

BATCH_TABLE = "batches"
PATIENT_COLUMN = "patient_name"
MEDICINES_COLUMN = "medicines_json"
# =========================================================


def split_medicine_names(text):
    """
    支持英文逗号和中文逗号分隔药品名称。
    """
    return [
        name.strip()
        for name in re.split(r"[,，]", text)
        if name.strip()
    ]


def build_medicines_json(medicine_names):
    """
    将药品名称列表转换为 JSON 字符串。
    """
    medicines = [
        {"medicine_name": name}
        for name in medicine_names
    ]

    return json.dumps(
        medicines,
        ensure_ascii=False
    )


def connect_db():
    """
    连接 MySQL 数据库。
    """
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset=DB_CHARSET,
    )


def insert_patient_batch(conn, patient_name, medicine_names):
    """
    插入一条病人药品记录。
    """
    medicines_json = build_medicines_json(medicine_names)

    sql = f"""
        INSERT INTO {BATCH_TABLE}
            ({PATIENT_COLUMN}, {MEDICINES_COLUMN})
        VALUES
            (%s, %s)
    """

    with conn.cursor() as cursor:
        cursor.execute(
            sql,
            (
                patient_name,
                medicines_json,
            ),
        )

    conn.commit()


def parse_args():
    """
    解析 Linux 命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="将病人及药品信息导入 MySQL 数据库"
    )

    parser.add_argument(
        "--patient",
        "-p",
        required=True,
        help="病人姓名，例如：张三",
    )

    parser.add_argument(
        "--medicines",
        "-m",
        required=True,
        help="药品名称，多个药品使用逗号分隔，例如：阿莫西林,布洛芬,维生素C",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    patient_name = args.patient.strip()
    medicine_names = split_medicine_names(args.medicines)

    # 参数校验
    if not patient_name:
        print("[错误] 病人姓名不能为空。")
        sys.exit(1)

    if not medicine_names:
        print("[错误] 至少需要输入一个药品名称。")
        sys.exit(1)

    conn = None

    try:
        print("正在连接数据库...")

        conn = connect_db()

        print("[完成] 数据库连接成功。")

        insert_patient_batch(
            conn,
            patient_name,
            medicine_names,
        )

        print(f"[完成] 已录入病人：{patient_name}")
        print(
            f"[完成] 已录入药品："
            f"{', '.join(medicine_names)}"
        )

    except pymysql.MySQLError as e:
        print(f"[错误] 数据库操作失败：{e}")
        sys.exit(1)

    except Exception as e:
        print(f"[错误] 程序执行失败：{e}")
        sys.exit(1)

    finally:
        if conn is not None:
            conn.close()
            print("[完成] 数据库连接已关闭。")


if __name__ == "__main__":
    main()