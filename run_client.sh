#!/usr/bin/env bash
# Client-side smoke test for ArtiFixer HTTP service:
#   health -> submit -> poll -> optional cancel
#
# Usage:
#   ./run_client.sh                  # submit + poll until done
#   ./run_client.sh cancel           # submit, then cancel immediately
#   ./run_client.sh cancel JOB_ID    # cancel an existing job
#   SERVER=http://172.18.1.203:8000 ./run_client.sh
#
set -euo pipefail

SERVER="${SERVER:-http://172.18.1.203:8000}"
API_KEY="${ARTIFIXER_API_KEY:-}"
SPLIT_PATH="${SPLIT_PATH:-/data/scenes/flowers_ds4_3view_prepared/split.json}"
RENDER_TRAJECTORY="${RENDER_TRAJECTORY:-val_frames}"
POLL_SEC="${POLL_SEC:-10}"

auth_headers=()
if [[ -n "$API_KEY" ]]; then
  auth_headers=(-H "X-API-Key: ${API_KEY}")
fi

json_field() {
  local json="$1"
  local key="$2"
  python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$key" <<<"$json"
}

echo "== health =="
curl -sS "${SERVER}/health"
echo
echo

if [[ "${1:-}" == "cancel" && -n "${2:-}" ]]; then
  JOB_ID="$2"
  echo "== cancel existing job ${JOB_ID} =="
  curl -sS -X POST "${SERVER}/v1/jobs/${JOB_ID}/cancel" "${auth_headers[@]}"
  echo
  echo
  echo "== job status =="
  curl -sS "${SERVER}/v1/jobs/${JOB_ID}" "${auth_headers[@]}"
  echo
  exit 0
fi

echo "== submit inference =="
SUBMIT_JSON=$(curl -sS -X POST "${SERVER}/v1/inference" \
  -H 'Content-Type: application/json' \
  "${auth_headers[@]}" \
  -d "{
    \"split_path\": \"${SPLIT_PATH}\",
    \"render_trajectory\": \"${RENDER_TRAJECTORY}\"
  }")
echo "$SUBMIT_JSON"
echo

JOB_ID=$(json_field "$SUBMIT_JSON" id)
STATUS=$(json_field "$SUBMIT_JSON" status)
echo "job_id=${JOB_ID} status=${STATUS}"
echo

if [[ "${1:-}" == "cancel" ]]; then
  echo "== cancel job ${JOB_ID} =="
  curl -sS -X POST "${SERVER}/v1/jobs/${JOB_ID}/cancel" "${auth_headers[@]}"
  echo
  echo
  echo "== job status after cancel =="
  curl -sS "${SERVER}/v1/jobs/${JOB_ID}" "${auth_headers[@]}"
  echo
  exit 0
fi

echo "== poll until finished (every ${POLL_SEC}s) =="
while true; do
  JOB_JSON=$(curl -sS "${SERVER}/v1/jobs/${JOB_ID}" "${auth_headers[@]}")
  STATUS=$(json_field "$JOB_JSON" status)
  echo "$(date -Is) status=${STATUS}"
  case "$STATUS" in
    succeeded|failed|cancelled)
      echo
      echo "$JOB_JSON"
      echo
      if [[ "$STATUS" == "succeeded" ]]; then
        echo "== download =="
        curl -fL "${SERVER}/v1/jobs/${JOB_ID}/download" \
          "${auth_headers[@]}" \
          -o "artifixer-${JOB_ID}.zip"
        echo "saved artifixer-${JOB_ID}.zip"
      fi
      exit 0
      ;;
  esac
  sleep "$POLL_SEC"
done
