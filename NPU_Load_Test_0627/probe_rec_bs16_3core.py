# -*- coding: utf-8 -*-
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rknnlite.api import RKNNLite


DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "model_ocr_bs16.rknn",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    cores = (
        RKNNLite.NPU_CORE_0,
        RKNNLite.NPU_CORE_1,
        RKNNLite.NPU_CORE_2,
    )
    workers = []
    try:
        for core in cores:
            rknn = RKNNLite()
            if rknn.load_rknn(model_path) != 0:
                raise RuntimeError("load_rknn failed on core mask {}".format(core))
            if rknn.init_runtime(core_mask=core) != 0:
                raise RuntimeError("init_runtime failed on core mask {}".format(core))
            workers.append((core, rknn))

        batch = np.zeros((16, 48, 320, 3), dtype=np.uint8)

        def infer(worker):
            core, rknn = worker
            started = time.perf_counter()
            outputs = rknn.inference(inputs=[batch])
            elapsed = time.perf_counter() - started
            if outputs is None:
                raise RuntimeError("inference returned None on core mask {}".format(core))
            return core, elapsed, [np.asarray(output).shape for output in outputs]

        wall_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(infer, workers))
        wall_elapsed = time.perf_counter() - wall_started

        print("model={}".format(model_path), flush=True)
        print("model_bytes={}".format(os.path.getsize(model_path)), flush=True)
        print("input_shape={}".format(batch.shape), flush=True)
        for core, elapsed, shapes in results:
            print(
                "core_mask={}: infer_ms={:.3f}, outputs={}".format(
                    core, elapsed * 1000.0, shapes
                ),
                flush=True,
            )
        print("parallel_wall_ms={:.3f}".format(wall_elapsed * 1000.0), flush=True)
        print("RESULT: PASS", flush=True)
        return 0
    finally:
        for _core, rknn in workers:
            rknn.release()


if __name__ == "__main__":
    raise SystemExit(main())
