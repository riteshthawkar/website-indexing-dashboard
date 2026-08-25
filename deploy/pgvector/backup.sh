#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PGVECTOR_ADMIN_DSN:-}" ]]; then
  echo "PGVECTOR_ADMIN_DSN is required" >&2
  exit 1
fi
if [[ "${PGVECTOR_ADMIN_DSN}" != *"sslmode="* ]]; then
  echo "PGVECTOR_ADMIN_DSN must specify sslmode" >&2
  exit 1
fi

backup_root="${PGVECTOR_BACKUP_DIR:-/var/backups/mbzuai-pgvector}"
install -d -m 0700 "${backup_root}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
destination="${backup_root}/mbzuai_vectors_${timestamp}.dump"

umask 077
pg_dump \
  --dbname="${PGVECTOR_ADMIN_DSN}" \
  --format=custom \
  --compress=9 \
  --no-owner \
  --file="${destination}"
sha256sum "${destination}" > "${destination}.sha256"
echo "${destination}"
