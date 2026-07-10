"""Interactively import patient medicine batches into MySQL.

Run this file directly. It connects to medicine_db.batches, asks for one
patient name and comma-separated medicine names, then inserts one row.
Input q at any prompt to exit without saving the current unfinished entry.
"""
import json
import re

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


QUIT_COMMAND = "q"


def is_quit(text):
    return text.strip().lower() == QUIT_COMMAND


def split_medicine_names(text):
    """Split medicine names by English or Chinese comma."""
    return [name.strip() for name in re.split(r"[,，]", text) if name.strip()]


def build_medicines_json(medicine_names):
    medicines = [{"medicine_name": name} for name in medicine_names]
    return json.dumps(medicines, ensure_ascii=False)


def connect_db():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset=DB_CHARSET,
    )


def insert_patient_batch(conn, patient_name, medicine_names):
    medicines_json = build_medicines_json(medicine_names)
    sql = f"""
        INSERT INTO {BATCH_TABLE}
            ({PATIENT_COLUMN}, {MEDICINES_COLUMN})
        VALUES
            (%s, %s)
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (patient_name, medicines_json))
    conn.commit()


def prompt_patient_batch():
    while True:
        patient_name = input("\n请输入要录入的病人姓名（输入 q 退出）：").strip()
        if is_quit(patient_name):
            return None
        if patient_name:
            break
        print("[提示] 病人姓名不能为空，请重新输入。")

    while True:
        medicines_text = input("请输入药品名称，多个药品用逗号分隔（输入 q 退出）：").strip()
        if is_quit(medicines_text):
            return None

        medicine_names = split_medicine_names(medicines_text)
        if medicine_names:
            return patient_name, medicine_names

        print("[提示] 至少需要输入一个药品名称，请重新输入。")


def main():
    print("正在连接数据库...")
    conn = connect_db()
    print("[完成] 数据库连接成功。")

    try:
        while True:
            patient_batch = prompt_patient_batch()
            if patient_batch is None:
                print("[退出] 程序已结束，当前未完成录入已作废。")
                break

            patient_name, medicine_names = patient_batch
            insert_patient_batch(conn, patient_name, medicine_names)
            print(f"[完成] 已录入病人：{patient_name}")
            print(f"[完成] 已录入药品：{', '.join(medicine_names)}")
    finally:
        conn.close()
        print("[完成] 数据库连接已关闭。")


if __name__ == "__main__":
    main()
