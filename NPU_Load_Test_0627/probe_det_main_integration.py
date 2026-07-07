# -*- coding: utf-8 -*-
import os
import sys

import numpy as np


PROJECT_ROOT = (
    "/home/forlinx/Models/AnotherYiliao/shibie/"
    "YiLiaoShiBie_0521/YiLiaoShiBie"
)


def main():
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    from src.identification.rknn_ocr_adapter import RknnOCRRecognizer

    recognizer = None
    try:
        recognizer = RknnOCRRecognizer(
            det_model_path=os.path.join(PROJECT_ROOT, "model_det_bs16.rknn"),
            rec_model_path=os.path.join(PROJECT_ROOT, "model_ocr_0624_bs32.rknn"),
            cls_model_path=os.path.join(PROJECT_ROOT, "model_cls_bs32.rknn"),
            det_input_size=448,
            det_batch_size=16,
            rec_batch_size=32,
            cls_batch_size=32,
        )
        print(
            "det_config: size={}, batch={}, workers={}".format(
                recognizer.det_input_size,
                recognizer.det_batch_size,
                [(worker["index"], worker["core"]) for worker in recognizer.det_workers],
            ),
            flush=True,
        )

        det_batch = np.zeros((16, 448, 448, 3), dtype=np.uint8)
        outputs, wall_time = recognizer._run_det_batch_jobs([
            {"job_index": 0, "det_batch": det_batch},
            {"job_index": 1, "det_batch": det_batch},
        ])
        shapes = [np.asarray(output[0]).shape for output in outputs]
        print(
            "dual_worker_result: workers_used={}, wall_ms={:.3f}, outputs={}".format(
                recognizer._last_det_workers,
                wall_time * 1000.0,
                shapes,
            ),
            flush=True,
        )
        print("RESULT: PASS", flush=True)
        return 0
    finally:
        if recognizer is not None:
            recognizer.release()


if __name__ == "__main__":
    sys.exit(main())
