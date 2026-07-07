# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time

import numpy as np
from rknnlite.api import RKNNLite


DEFAULT_PROJECT_ROOT = (
    "/home/forlinx/Models/AnotherYiliao/shibie/"
    "YiLiaoShiBie_0521/YiLiaoShiBie"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strictly verify model_det_bs16.rknn with a 16x448x448x3 input."
    )
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--model", default="model_det_bs16.rknn")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(args.project_root, model_path)
    model_path = os.path.abspath(model_path)

    if not os.path.isfile(model_path):
        print("ERROR: model not found: {}".format(model_path), flush=True)
        return 2

    test_input = np.zeros((16, 448, 448, 3), dtype=np.uint8)
    print("model={}".format(model_path), flush=True)
    print("model_bytes={}".format(os.path.getsize(model_path)), flush=True)
    print("input_shape={}".format(test_input.shape), flush=True)
    print("input_dtype={}".format(test_input.dtype), flush=True)

    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(model_path)
        print("load_rknn_ret={}".format(ret), flush=True)
        if ret != 0:
            return 3

        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        print("init_runtime_ret={}".format(ret), flush=True)
        if ret != 0:
            return 4

        started = time.perf_counter()
        outputs = rknn.inference(inputs=[test_input])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if outputs is None:
            print("ERROR: inference returned None", flush=True)
            return 5

        print("inference_ms={:.3f}".format(elapsed_ms), flush=True)
        print("output_count={}".format(len(outputs)), flush=True)
        for index, output in enumerate(outputs):
            array = np.asarray(output)
            print(
                "output_{}: shape={}, dtype={}".format(
                    index, array.shape, array.dtype
                ),
                flush=True,
            )
        print("RESULT: PASS - model accepts strict batch16 448x448 RGB-shaped input", flush=True)
        return 0
    except Exception as exc:
        print("RESULT: FAIL - {}: {}".format(type(exc).__name__, exc), flush=True)
        return 1
    finally:
        rknn.release()


if __name__ == "__main__":
    sys.exit(main())
