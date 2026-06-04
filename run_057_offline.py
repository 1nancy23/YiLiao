from pathlib import Path

from detect_image import build_parser, run


ROOT_DIR = Path(__file__).resolve().parent


def main():
    parser = build_parser()
    args = parser.parse_args(
        [
            str(ROOT_DIR / "057.png"),
            "--output-json",
            str(ROOT_DIR / "offline_057_full.json"),
            "--output-image",
            str(ROOT_DIR / "offline_057_full.jpg"),
        ]
    )
    run(args)


if __name__ == "__main__":
    main()
