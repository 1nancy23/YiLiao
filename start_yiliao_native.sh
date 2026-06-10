#!/usr/bin/env bash
set -u

PROJECT_DIR="/home/forlinx/Models/AnotherYiliao/shibie/YiLiaoShiBie_0521/YiLiaoShiBie"
PYTHON_BIN="/home/forlinx/Models/Python2/bin/python3.9"
LOG_FILE="/home/forlinx/yiliao_native_start.log"

if [ -z "${DISPLAY:-}" ]; then
  export DISPLAY=":0"
fi

if [ -z "${XAUTHORITY:-}" ]; then
  if [ -f "/run/user/1000/gdm/Xauthority" ]; then
    export XAUTHORITY="/run/user/1000/gdm/Xauthority"
  elif [ -f "/home/forlinx/.Xauthority" ]; then
    export XAUTHORITY="/home/forlinx/.Xauthority"
  fi
fi

export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
export YILIAO_RUNTIME_LOGS="${YILIAO_RUNTIME_LOGS:-0}"
export YILIAO_VERBOSE_RUNTIME="${YILIAO_VERBOSE_RUNTIME:-0}"
export YILIAO_QUIET_OCR="${YILIAO_QUIET_OCR:-0}"
export YILIAO_COLLECT_TIMING="${YILIAO_COLLECT_TIMING:-0}"

: > "${LOG_FILE}"

{
  echo "============================================================"
  date '+%Y-%m-%d %H:%M:%S'
  echo "Starting YiLiao native recognition UI"
  echo "DISPLAY=${DISPLAY:-}"
  echo "XAUTHORITY=${XAUTHORITY:-}"
  echo "YILIAO_RUNTIME_LOGS=${YILIAO_RUNTIME_LOGS}"
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "ARGS=$*"
} >> "${LOG_FILE}" 2>&1

cd "${PROJECT_DIR}" || {
  echo "Project directory not found: ${PROJECT_DIR}" >> "${LOG_FILE}" 2>&1
  exit 1
}

exec "${PYTHON_BIN}" main.py "$@" >> "${LOG_FILE}" 2>&1
