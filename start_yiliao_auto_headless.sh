#!/usr/bin/env bash
set -u

PROJECT_DIR="/home/forlinx/Models/AnotherYiliao/shibie/YiLiaoShiBie_0521/YiLiaoShiBie"
PYTHON_BIN="/home/forlinx/Models/Python2/bin/python3.9"
LOG_FILE="/home/forlinx/yiliao_auto_headless.log"
PID_FILE="/tmp/yiliao_auto_headless.pid"

export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
export YILIAO_HEADLESS="1"
export YILIAO_RUNTIME_LOGS="${YILIAO_RUNTIME_LOGS:-0}"
export YILIAO_VERBOSE_RUNTIME="${YILIAO_VERBOSE_RUNTIME:-0}"
export YILIAO_QUIET_OCR="${YILIAO_QUIET_OCR:-1}"
export YILIAO_COLLECT_TIMING="${YILIAO_COLLECT_TIMING:-0}"

if [ -f "${PID_FILE}" ]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') already running pid=${old_pid}" >> "${LOG_FILE}"
    exit 0
  fi
fi

cd "${PROJECT_DIR}" || {
  echo "$(date '+%Y-%m-%d %H:%M:%S') missing project dir: ${PROJECT_DIR}" >> "${LOG_FILE}"
  exit 1
}

echo $$ > "${PID_FILE}"
{
  echo "============================================================"
  date '+%Y-%m-%d %H:%M:%S'
  echo "Starting YiLiao auto headless recognition"
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "ARGS=$*"
} >> "${LOG_FILE}" 2>&1

exec "${PYTHON_BIN}" auto_headless_app.py "$@" >> "${LOG_FILE}" 2>&1
