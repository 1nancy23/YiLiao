import json
import os

import pymysql
import yaml

from run_realtime_detection_yolo_new_3 import run_realtime_detection
from src.identification.DrugMatcher import DrugMatcher
from src.identification.Recog import PharmaceuticalBottleClassifier
from src.identification.rknn_ocr_adapter import RknnOCRRecognizer


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db = config.get("db_config", {})
    kwargs = dict(
        host=db.get("host", "192.168.137.1"),
        user=db.get("user", "root"),
        password=db.get("password", "root"),
        database=db.get("database", "medicine_db"),
        charset=db.get("charset", "utf8"),
        port=int(db.get("port", 3306)),
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        conn = pymysql.connect(**kwargs)
    except pymysql.err.OperationalError as exc:
        if exc.args and exc.args[0] == 1115 and kwargs["charset"].lower() == "utf8mb4":
            kwargs["charset"] = "utf8"
            conn = pymysql.connect(**kwargs)
        else:
            raise

    tables = config["table_config"]
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(1) AS n FROM {tables['drug_table']}")
        drug_count = cursor.fetchone()
        cursor.execute(f"SELECT COUNT(1) AS n FROM {tables['patient_table']}")
        patient_count = cursor.fetchone()
    print("db_connected", conn.open)
    print("drug_count", drug_count)
    print("patient_count", patient_count)

    matcher = DrugMatcher(
        conn,
        drug_table=tables["drug_table"],
        drug_column=tables["drug_column"],
        patient_table=tables["patient_table"],
        patient_column=tables["patient_column"],
        cache_drugs=True,
    )
    os.environ.setdefault("YILIAO_FEATURE_CACHE", "./single_image_feature_cache.pkl")
    classifier = PharmaceuticalBottleClassifier(db_conn=conn, device="npu")
    ocrs = [RknnOCRRecognizer() for _ in range(3)]
    result = run_realtime_detection(
        model=None,
        ocr_recognizer=ocrs,
        drug_matcher=matcher,
        classifier=classifier,
        device="npu",
        single_image_path="005.png",
        single_image_output_json="./realtime_single_db_sift/result.json",
        recognition_workers=3,
        classifier_thread_safe=False,
        quiet_ocr=True,
    )

    bg = (result or {}).get("background") or {}
    print("runtime", (result or {}).get("runtime_sec_excluding_model_init"))
    print("counts", bg.get("counts"))
    for item in sorted(bg.get("results", []), key=lambda x: (x.get("type"), x.get("index"))):
        if item.get("type") == "bottle":
            print(
                "bottle",
                item.get("index"),
                item.get("final_medicine"),
                item.get("classification_method"),
                item.get("confidence"),
                item.get("sift_best_template"),
                item.get("candidates", [])[:3],
            )
        elif item.get("type") == "bag":
            print("bag", item.get("index"), item.get("patient_name"), item.get("ocr_text"))
        else:
            print("shuye", item.get("index"), item.get("status"), item.get("liquid"), item.get("concentration"), item.get("volume"))

    conn.close()


if __name__ == "__main__":
    main()
