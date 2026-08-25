#!/usr/bin/env bash
set -euo pipefail

reader_password_file="${PGVECTOR_READER_PASSWORD_FILE:-}"
writer_password_file="${PGVECTOR_WRITER_PASSWORD_FILE:-}"

if [[ -z "${reader_password_file}" || ! -r "${reader_password_file}" ]]; then
  echo "PGVECTOR_READER_PASSWORD_FILE must point to a readable Docker secret" >&2
  exit 1
fi
if [[ -z "${writer_password_file}" || ! -r "${writer_password_file}" ]]; then
  echo "PGVECTOR_WRITER_PASSWORD_FILE must point to a readable Docker secret" >&2
  exit 1
fi

reader_password="$(<"${reader_password_file}")"
writer_password="$(<"${writer_password_file}")"

if [[ ${#reader_password} -lt 32 || ${#writer_password} -lt 32 ]]; then
  echo "pgvector login-role passwords must contain at least 32 characters" >&2
  exit 1
fi

psql \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set ON_ERROR_STOP=1 \
  --set reader_user="${PGVECTOR_READER_USER}" \
  --set reader_password="${reader_password}" \
  --set reader_limit="${PGVECTOR_READER_CONNECTION_LIMIT:-24}" \
  --set writer_user="${PGVECTOR_WRITER_USER}" \
  --set writer_password="${writer_password}" \
  --set writer_limit="${PGVECTOR_WRITER_CONNECTION_LIMIT:-8}" <<'SQL'
SELECT format(
    'CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION',
    :'reader_user', :'reader_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'reader_user') \gexec
SELECT format('ALTER ROLE %I PASSWORD %L CONNECTION LIMIT %s', :'reader_user', :'reader_password', :'reader_limit') \gexec
SELECT format('GRANT mbzuai_retrieval_reader TO %I', :'reader_user') \gexec
SELECT format('ALTER ROLE %I SET default_transaction_read_only = on', :'reader_user') \gexec
SELECT format('ALTER ROLE %I SET statement_timeout = %L', :'reader_user', '10s') \gexec
SELECT format('ALTER ROLE %I SET idle_in_transaction_session_timeout = %L', :'reader_user', '5s') \gexec

SELECT format(
    'CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION',
    :'writer_user', :'writer_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'writer_user') \gexec
SELECT format('ALTER ROLE %I PASSWORD %L CONNECTION LIMIT %s', :'writer_user', :'writer_password', :'writer_limit') \gexec
SELECT format('GRANT mbzuai_retrieval_writer TO %I', :'writer_user') \gexec
SELECT format('ALTER ROLE %I SET statement_timeout = %L', :'writer_user', '120s') \gexec
SELECT format('ALTER ROLE %I SET idle_in_transaction_session_timeout = %L', :'writer_user', '15s') \gexec
SQL

unset reader_password writer_password
