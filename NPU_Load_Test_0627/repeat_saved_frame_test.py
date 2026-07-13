# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

import yaml


def compact_result(run_index, result):
    background = (result or {}).get("background") or {}
    return {
        "run": run_index,
        "counts": background.get("counts"),
        "patient_name": background.get("patient_name"),
        "bags": background.get("bags") or [],
        "recognized_medicines": background.get("recognized_medicines") or [],
        "database_match": background.get("database_match") or {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    sys.path.insert(0, project_root)
    os.environ.setdefault("YILIAO_RUNTIME_LOGS", "0")
    os.environ.setdefault("YILIAO_QUIET_OCR", "1")

    from native_app import create_local_bottle_db_matcher, init_db, release_models
    from run_realtime_detection_yolo_new_3 import run_realtime_detection
    from src.identification.DrugMatcher import DrugMatcher
    from src.identification.Recog import PharmaceuticalBottleClassifier
    from src.identification.rknn_ocr_adapter import RknnOCRRecognizer

    with open("config.yaml", "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    conn = None
    classifier = None
    recognizers = []
    try:
        classifier = PharmaceuticalBottleClassifier(db_conn=None, device="npu")
        conn = init_db(config.get("db_config", {}))
        tables = config["table_config"]
        matcher = create_local_bottle_db_matcher(
            DrugMatcher,
            conn,
            classifier.get_cached_names(),
            drug_table=tables["drug_table"],
            drug_column=tables["drug_column"],
            patient_table=tables["patient_table"],
            patient_column=tables["patient_column"],
            batch_table=tables.get("batch_table", "batches"),
            batch_medicines_column=tables.get("batch_medicines_column", "medicines_json"),
        )
        recognizers = [RknnOCRRecognizer(
            det_model_path=os.path.join(project_root, "model_det_bs16.rknn"),
            rec_model_path=os.path.join(project_root, "model_ocr_bs16.rknn"),
            cls_model_path=os.path.join(project_root, "model_cls_bs32.rknn"),
            det_input_size=448,
            det_batch_size=16,
            rec_batch_size=16,
            cls_batch_size=32,
        )]

        summaries = []
        for index in range(1, max(1, args.runs) + 1):
            result = run_realtime_detection(
                ocr_recognizer=recognizers,
                drug_matcher=matcher,
                classifier=classifier,
                classifier_thread_safe=False,
                yolo_model_path=config["model"].get("yolo_rknn_path", "./model_yolo_0615.rknn"),
                yolo_input_size=int(config["model"].get("yolo_input_size", 640)),
                single_image_path=os.path.abspath(args.image),
                quiet_ocr=True,
                headless=True,
            )
            summary = compact_result(index, result)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

        print("SUMMARY=" + json.dumps(summaries, ensure_ascii=False), flush=True)
    finally:
        release_models(classifier, recognizers)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
