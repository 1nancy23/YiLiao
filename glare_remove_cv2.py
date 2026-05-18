import argparse
import os

import cv2

from src.utils.img_utils import ImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Reduce reflective glare with OpenCV.")
    parser.add_argument("--input", default="image6.png", help="Input image path.")
    parser.add_argument("--output", default="image6_glare_reduced.png", help="Output image path.")
    parser.add_argument("--mask-output", default="image6_glare_mask.png", help="Output glare mask path.")
    parser.add_argument("--strength", type=float, default=0.85, help="Inpaint blend strength, 0.0-1.0.")
    parser.add_argument("--radius", type=int, default=5, help="OpenCV inpaint radius.")
    args = parser.parse_args()

    result, mask = ImageProcessor.reduce_glare(
        args.input,
        strength=args.strength,
        inpaint_radius=args.radius,
        return_mask=True,
    )
    if result is None:
        raise SystemExit(f"Failed to process: {args.input}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, result)
    cv2.imwrite(args.mask_output, mask)
    print(f"saved: {args.output}")
    print(f"saved: {args.mask_output}")


if __name__ == "__main__":
    main()
