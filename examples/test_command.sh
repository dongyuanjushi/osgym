curl -sS -X POST 'http://127.0.0.1:30001/reset' \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d '{
    "task_config": {},
    "timeout": 600
  }'