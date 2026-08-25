#!/usr/bin/env bash
set -euo pipefail

allowed_cidrs="${PGVECTOR_ALLOWED_CIDRS:-}"
if [[ -z "${allowed_cidrs}" ]]; then
  echo "PGVECTOR_ALLOWED_CIDRS must contain one or more private client CIDRs" >&2
  exit 1
fi

hba_tmp="$(mktemp)"
trap 'rm -f "${hba_tmp}"' EXIT
{
  echo "local all all peer"
  IFS=',' read -r -a cidrs <<<"${allowed_cidrs}"
  for raw_cidr in "${cidrs[@]}"; do
    cidr="${raw_cidr//[[:space:]]/}"
    if [[ ! "${cidr}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}/([0-9]|[12][0-9]|3[0-2])$ ]] \
      && [[ ! "${cidr}" =~ ^[0-9a-fA-F:]+/([0-9]|[1-9][0-9]|1[01][0-9]|12[0-8])$ ]]; then
      echo "Invalid CIDR in PGVECTOR_ALLOWED_CIDRS" >&2
      exit 1
    fi
    normalized_cidr="$(
      psql --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" \
        --set ON_ERROR_STOP=1 --set cidr="${cidr}" --tuples-only --no-align \
        --command "SELECT :'cidr'::cidr"
    )"
    printf 'hostssl all all %s scram-sha-256\n' "${normalized_cidr}"
  done
  echo "hostnossl all all 0.0.0.0/0 reject"
  echo "hostnossl all all ::0/0 reject"
} >"${hba_tmp}"

install -o postgres -g postgres -m 0600 "${hba_tmp}" "${PGDATA}/pg_hba.conf"
