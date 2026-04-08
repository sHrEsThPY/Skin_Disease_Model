#!/bin/bash
status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ "$status_code" -eq 200 ]; then
  echo "Health check passed"
  exit 0
else
  echo "Health check failed with status: $status_code"
  exit 1
fi
