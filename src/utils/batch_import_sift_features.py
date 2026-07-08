"""Batch import medicine SIFT features into MySQL.

Run this file directly after the two-table database has been created.
It reads medicine template images from FEATURE_ROOT and writes SIFT blobs
to medicine_db.drugs.sift1 ... sift6.
"""
import os
import sys

import pymysql


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.identification.Recog import PharmaceuticalBottleClassifier


# ==================== Editable config ====================
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = "root"
DB_NAME = "medicine_db"
DB_CHARSET = "utf8mb4"

FEATURE_ROOT = r"D:\A_Python\工业图像识别\src\identification\feat_data"
# =========================================================


def main():
    if not os.path.isdir(FEATURE_ROOT):
        raise FileNotFoundError(f"Feature folder does not exist: {FEATURE_ROOT}")

    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset=DB_CHARSET,
    )

    try:
        classifier = PharmaceuticalBottleClassifier(conn, device="cpu")
        classifier.save_batch_features_to_db(FEATURE_ROOT)
        print(f"[完成] 药品 SIFT 特征已批量录入: {FEATURE_ROOT}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
