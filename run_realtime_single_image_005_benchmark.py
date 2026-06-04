import argparse
import os
from threading import Lock

import yaml

from run_single_image_057_fast import (
    init_fast_ocr,
    load_or_build_name_cache,
    init_db_from_config,
)
from run_realtime_detection_yolo_new_3 import run_realtime_detection
from src.identification.DrugMatcher import DrugMatcher
from src.identification.OCRRecognizer import OCRRecognizer_ori
from src.identification.Recog import PharmaceuticalBottleClassifier


class MatcherAdapter:
    def __init__(self, cached_matcher, db_matcher=None):
        self.cached_matcher = cached_matcher
        self.db_matcher = db_matcher

    def match(self, *args, **kwargs):
        return self.cached_matcher.match(*args, **kwargs)

    def check_patient_batch_medicines(self, *args, **kwargs):
        if self.db_matcher is not None:
            return self.db_matcher.check_patient_batch_medicines(*args, **kwargs)
        return {
            "batch_exists": False,
            "matched": False,
            "actual": [],
            "missing": kwargs.get("expected_medicine_names", []),
            "extra": [],
        }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run run_realtime_detection_yolo_new_3.py logic on 005.png without RTSP."
    )
    parser.add_argument("--image", default="005.png")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ocr-instances", type=int, default=12)
    parser.add_argument("--ocr-threads", type=int, default=4)
    parser.add_argument("--ocr-batch", type=int, default=8)
    parser.add_argument("--name-cache", default="./single_image_name_cache.json")
    parser.add_argument("--feature-cache", default="./single_image_feature_cache.pkl")
    parser.add_argument("--output-json", default="./realtime_single_005_result.json")
    return parser


def main():
    args = build_parser().parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cached_matcher, conn = load_or_build_name_cache(config, args.name_cache)
    db_matcher = None
    if conn is None:
        try:
            conn = init_db_from_config(config)
        except Exception:
            conn = None

    if conn is not None:
        tables = config["table_config"]
        db_matcher = DrugMatcher(
            conn,
            drug_table=tables["drug_table"],
            drug_column=tables["drug_column"],
            patient_table=tables["patient_table"],
            patient_column=tables["patient_column"],
            cache_drugs=True,
        )

    os.environ["YILIAO_FEATURE_CACHE"] = args.feature_cache
    classifier = PharmaceuticalBottleClassifier(db_conn=conn, device="npu")

    recognizers = []
    for _ in range(args.ocr_instances):
        recognizers.append(
            OCRRecognizer_ori(
                init_fast_ocr(
                    cpu_threads=args.ocr_threads,
                    rec_batch_num=args.ocr_batch,
                    show_log=False,
                )
            )
        )

    # Keep a reference to locks in this process for symmetry with the realtime flow.
    _locks = [Lock() for _ in recognizers]

    run_realtime_detection(
        model=None,
        checkpoint_path=config["model"]["checkpoint_path"],
        num_classes=config["model"]["num_classes"],
        ocr_recognizer=recognizers,
        drug_matcher=MatcherAdapter(cached_matcher, db_matcher),
        classifier=classifier,
        length=3,
        tile_size=config["segmentor"]["tile_size"],
        overlap=config["segmentor"]["overlap"],
        target_fps=config["segmentor"]["target_fps"],
        batch_frames=config["segmentor"]["batch_frames"],
        output_type=config["display"]["output_type"],
        overlay_alpha=config["display"]["overlay_alpha"],
        display_scale=config["display"]["display_scale"],
        save_video=False,
        device="npu",
        trigger_interval=0,
        single_image_path=args.image,
        single_image_output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
