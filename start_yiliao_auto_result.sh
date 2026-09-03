#!/usr/bin/env bash
set -u

PROJECT_DIR="/home/forlinx/Models/AnotherYiliao/shibie/YiLiaoShiBie_0521/YiLiaoShiBie"
PYTHON_BIN="/home/forlinx/Models/Python2/bin/python3.9"
LOG_FILE="/home/forlinx/yiliao_auto_result.log"
PID_FILE="/tmp/yiliao_auto_result.pid"

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
export YILIAO_TRIGGER_MODE="auto"
export YILIAO_RUNTIME_LOGS="${YILIAO_RUNTIME_LOGS:-0}"
export YILIAO_VERBOSE_RUNTIME="${YILIAO_VERBOSE_RUNTIME:-0}"
export YILIAO_QUIET_OCR="${YILIAO_QUIET_OCR:-0}"
export YILIAO_COLLECT_TIMING="${YILIAO_COLLECT_TIMING:-0}"

if [ -f "${PID_FILE}" ]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') already running pid=${old_pid}" >> "${LOG_FILE}"
    exit 0
  fi
fi

: > "${LOG_FILE}"
{
  echo "============================================================"
  date '+%Y-%m-%d %H:%M:%S'
  echo "Starting YiLiao automatic result UI"
  echo "DISPLAY=${DISPLAY:-}"
  echo "XAUTHORITY=${XAUTHORITY:-}"
  echo "PROJECT_DIR=${PROJECT_DIR}"
} >> "${LOG_FILE}" 2>&1

cd "${PROJECT_DIR}" || {
  echo "Project directory not found: ${PROJECT_DIR}" >> "${LOG_FILE}" 2>&1
  exit 1
}

echo $$ > "${PID_FILE}"
exec "${PYTHON_BIN}" native_app.py --auto --result-only --fullscreen >> "${LOG_FILE}" 2>&1
