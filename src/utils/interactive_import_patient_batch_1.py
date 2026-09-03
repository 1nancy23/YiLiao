#!/usr/bin/env python3
"""通过 Linux 命令行向 MySQL 录入一条病人用药批次。

推荐在项目根目录直接运行：

    python3 src/utils/interactive_import_patient_batch.py \
        --patient "张三" \
        --medicines "注射用头孢他啶,葡萄糖注射液"

也可以多次传入 ``--medicines``：

    python3 src/utils/interactive_import_patient_batch.py \
        -p "张三" -m "注射用头孢他啶" -m "葡萄糖注射液"

数据库连接默认读取项目根目录的 config.yaml；命令行参数和环境变量可覆盖配置。
使用 ``--dry-run`` 可以只检查输入和生成的 JSON，不连接数据库。
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pymysql
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def split_medicine_names(values):
    """拆分中英文逗号，并在保留顺序的同时去重。"""
    names = []
    seen = set()
    for value in values or []:
        for name in re.split(r"[,，]", value):
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def build_medicines_json(medicine_names):
    medicines = [{"medicine_name": name} for name in medicine_names]
    return json.dumps(medicines, ensure_ascii=False, separators=(",", ":"))


def load_project_config(config_path):
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config, path


def validate_identifier(value, label):
    value = str(value or "").strip()
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{label}不是合法的 MySQL 标识符：{value!r}")
    return value


def resolve_settings(args, config):
    db_config = config.get("db_config", {}) or {}
    table_config = config.get("table_config", {}) or {}

    settings = {
        "host": args.db_host or os.environ.get("YILIAO_DB_HOST") or db_config.get("host", "127.0.0.1"),
        "port": args.db_port or int(os.environ.get("YILIAO_DB_PORT", db_config.get("port", 3306))),
        "user": args.db_user or os.environ.get("YILIAO_DB_USER") or db_config.get("user", "root"),
        "password": (
            args.db_password
            if args.db_password is not None
            else os.environ.get("YILIAO_DB_PASSWORD", db_config.get("password", ""))
        ),
        "database": args.db_name or os.environ.get("YILIAO_DB_NAME") or db_config.get("database", "medicine_db2"),
        "charset": args.db_charset or os.environ.get("YILIAO_DB_CHARSET") or db_config.get("charset", "utf8"),
        "batch_table": table_config.get("batch_table", "batches"),
        "patient_column": table_config.get("patient_column", "patient_name"),
        "medicines_column": table_config.get("batch_medicines_column", "medicines_json"),
    }

    settings["batch_table"] = validate_identifier(settings["batch_table"], "批次表名")
    settings["patient_column"] = validate_identifier(settings["patient_column"], "病人姓名列名")
    settings["medicines_column"] = validate_identifier(settings["medicines_column"], "药品 JSON 列名")
    return settings


def connect_db(settings):
    return pymysql.connect(
        host=settings["host"],
        port=int(settings["port"]),
        user=settings["user"],
        password=settings["password"],
        database=settings["database"],
        charset=settings["charset"],
    )


def insert_patient_batch(conn, settings, patient, medicine_names, optional_fields):
    columns = [settings["patient_column"], settings["medicines_column"]]
    values = [patient, build_medicines_json(medicine_names)]

    for column, value in optional_fields:
        if value is not None:
            columns.append(validate_identifier(column, "病人信息列名"))
            values.append(value)

    placeholders = ", ".join(["%s"] * len(values))
    sql = (
        f"INSERT INTO {settings['batch_table']} "
        f"({', '.join(columns)}) VALUES ({placeholders})"
    )
    with conn.cursor() as cursor:
        cursor.execute(sql, values)
        batch_id = cursor.lastrowid
    conn.commit()
    return batch_id


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="通过 Linux 命令行向 MySQL 录入病人及其用药信息。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patient", "-p", required=True, help="病人姓名")
    parser.add_argument(
        "--medicines",
        "-m",
        required=True,
        action="append",
        help="药品名称；可用中英文逗号分隔，也可重复传入该参数",
    )
    parser.add_argument("--gender", help="病人性别，可选")
    parser.add_argument("--age", type=int, help="病人年龄，可选")
    parser.add_argument("--department", help="科室，可选")
    parser.add_argument("--bed-no", help="床号，可选")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="项目 YAML 配置文件")
    parser.add_argument("--db-host", help="覆盖数据库地址")
    parser.add_argument("--db-port", type=int, help="覆盖数据库端口")
    parser.add_argument("--db-user", help="覆盖数据库用户名")
    parser.add_argument("--db-password", help="覆盖数据库密码；更推荐使用 YILIAO_DB_PASSWORD")
    parser.add_argument("--db-name", help="覆盖数据库名称")
    parser.add_argument("--db-charset", help="覆盖数据库字符集")
    parser.add_argument("--dry-run", action="store_true", help="只预览录入内容，不连接数据库")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    patient = args.patient.strip()
    medicine_names = split_medicine_names(args.medicines)

    if not patient:
        print("[错误] 病人姓名不能为空。", file=sys.stderr)
        return 2
    if not medicine_names:
        print("[错误] 至少需要提供一个药品名称。", file=sys.stderr)
        return 2
    if args.age is not None and not 0 <= args.age <= 150:
        print("[错误] 病人年龄必须在 0～150 之间。", file=sys.stderr)
        return 2

    conn = None
    try:
        config, config_path = load_project_config(args.config)
        settings = resolve_settings(args, config)
        medicines_json = build_medicines_json(medicine_names)

        print(f"[配置] {config_path}")
        print(f"[病人] {patient}")
        print(f"[药品] {', '.join(medicine_names)}")
        print(f"[JSON] {medicines_json}")

        if args.dry_run:
            print("[预览完成] --dry-run 模式未连接数据库，也未写入数据。")
            return 0

        conn = connect_db(settings)
        batch_id = insert_patient_batch(
            conn,
            settings,
            patient,
            medicine_names,
            [
                ("patient_gender", args.gender),
                ("patient_age", args.age),
                ("department", args.department),
                ("bed_no", args.bed_no),
            ],
        )
        print(f"[完成] 病人用药批次已写入，batch_id={batch_id}")
        return 0
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"[错误] 参数或配置无效：{error}", file=sys.stderr)
        return 2
    except pymysql.MySQLError as error:
        if conn is not None:
            conn.rollback()
        print(f"[错误] 数据库操作失败：{error}", file=sys.stderr)
        return 1
    except Exception as error:
        if conn is not None:
            conn.rollback()
        print(f"[错误] 执行失败：{error}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
