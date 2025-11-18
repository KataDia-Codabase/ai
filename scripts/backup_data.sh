#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '\n')
fi

BACKUP_ROOT=${BACKUP_ROOT:-backups}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MYSQL_PASSWORD_VALUE=${MYSQL_ROOT_PASSWORD:-rootpassword}

mkdir -p "$BACKUP_ROOT/mysql" "$BACKUP_ROOT/redis"

if docker ps --format '{{.Names}}' | grep -q '^katadia-mysql$'; then
  docker exec katadia-mysql mysqldump -u root -p"$MYSQL_PASSWORD_VALUE" --databases "${MYSQL_DATABASE:-katadia_ml}" \
    > "$BACKUP_ROOT/mysql/${MYSQL_DATABASE:-katadia_ml}_$TIMESTAMP.sql"
else
  echo "[ERROR] katadia-mysql container not running" >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -q '^katadia-redis$'; then
  docker exec katadia-redis redis-cli save >/dev/null
  docker cp katadia-redis:/data "$BACKUP_ROOT/redis/data_$TIMESTAMP"
else
  echo "[ERROR] katadia-redis container not running" >&2
  exit 1
fi

echo "Backups stored in $BACKUP_ROOT (timestamp $TIMESTAMP)"
