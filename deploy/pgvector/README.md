# MBZUAI pgvector production deployment

This directory runs PostgreSQL 17 with pgvector 0.8.6 on a DigitalOcean
Droplet. The database is private infrastructure: it must not have a public
PostgreSQL listener or a public firewall rule.

## Production shape

- Start with a 2-vCPU / 4-GB Droplet and a separately mounted Block Storage
  Volume. The current corpus is small enough that its 1,536-dimensional float
  vectors and HNSW graph should remain comfortably inside this memory budget;
  confirm with the real load test before downsizing the chatbot container.
- Put the Droplet and App Platform app in the same VPC. Bind Compose to the
  Droplet's private VPC address and allow TCP 5432 only from the app/build
  worker private ranges.
- Use a private DNS name whose certificate SAN matches the pgvector host. The
  runtime DSN should use `sslmode=verify-full`; `sslmode=require` is an explicit
  fallback, not the preferred setting.
- Keep `/mnt/mbzuai-pgvector/postgres` on the mounted Volume. Database files
  must never live only on the Droplet root disk.

The schema supports eleven typed retrieval lanes under an immutable
`release_id`, including the evaluated `page_cards` and `actions` navigation
lanes.
An HNSW cosine index serves dense retrieval. B-tree indexes support release,
lane, URL, and language filters; JSONB and multilingual `simple` full-text
indexes are already present for a later controlled hybrid-search experiment.

## First bootstrap

1. Install Docker Engine and the Compose plugin from Docker's repository.
2. Mount and persist the Volume before creating `PGVECTOR_DATA_DIR`.
3. Install a certificate chain as `server.crt`, `server.key`, and `ca.crt` in
   `PGVECTOR_TLS_DIR`. PostgreSQL requires `server.key` to be mode `0600` and
   readable by the container's postgres user (UID 999 in the pinned image).
4. Create three independent random secrets of at least 32 characters in root-
   owned mode-`0600` files: admin, reader, and writer passwords.
5. Copy `.env.example` to `.env`, set the private IP and absolute host paths,
   and validate before starting:

```bash
docker compose --env-file .env config --quiet
docker compose --env-file .env up -d
docker compose --env-file .env ps
docker compose --env-file .env logs --tail=100 postgres
```

The init scripts create the extension, schema, constraints, indexes, group
roles, and least-privilege login roles only when `PGDATA` is empty. For an
existing cluster, apply numbered migrations explicitly with the admin DSN and
record the change window; do not erase the Volume to rerun initialization.

Schema v2 adds the selected-release Page Card and action lanes. Apply it to an
existing cluster during a recorded change window before indexing:

```bash
psql "$PGVECTOR_ADMIN_DSN" \
  --set ON_ERROR_STOP=1 \
  --file deploy/pgvector/sql/004_selected_release_lanes.sql
```

## Application credentials

The retriever receives only the reader DSN in `PGVECTOR_DSN`:

```text
postgresql://mbzuai_retriever:<url-encoded-password>@pgvector.internal:5432/mbzuai_vectors?sslmode=verify-full&sslrootcert=/path/to/ca.crt
```

The indexing worker receives the separate writer DSN in
`PGVECTOR_INGEST_DSN`. Never put that variable on the public backend or the
long-lived retriever. The committed App Platform spec contains only the reader
secret placeholder.

The runtime pool is bounded at 12 connections and the writer at 8. With one
retriever instance, the configured 100 server connections retain ample room
for PostgreSQL maintenance and operations. Recalculate this budget before
adding replicas: `instances × pool_max + writers + admin headroom` must stay
below `max_connections`.

## Index and release lifecycle

The `gemini_pgvector` pipeline stage:

1. creates or resumes a `building` candidate tied to one immutable indexing
   contract;
2. embeds and commits small batches independently;
3. verifies exact namespace counts; and
4. marks the candidate `ready`.

It never promotes the candidate automatically. After release evaluation and
the immutable active-pointer promotion, operations may record the database
activation atomically per project:

```bash
PGVECTOR_INGEST_DSN='...' \
python scripts/pgvector_admin.py activate \
  --config mbzuai_production \
  --work-dir /data/releases/runs/mbzuai_main/<run-id>
```

The active release pointer remains the serving authority. Ready and retired
rows remain queryable by their exact immutable release ID during blue-green
draining, so a database status update cannot interrupt the old process.

Run a non-mutating database and cardinality check with:

```bash
PGVECTOR_DSN='...' \
python scripts/pgvector_admin.py health \
  --config mbzuai_production \
  --work-dir /data/releases/runs/mbzuai_main/<run-id>
```

## Backups, monitoring, and maintenance

- Schedule `backup.sh` with an admin DSN, encrypt and copy each dump off the
  Droplet, and alert on missing backups. A Volume or Droplet snapshot is not a
  substitute for tested logical backups.
- Perform a restore drill into an isolated PostgreSQL instance at least
  quarterly, then verify schema version, release counts, and a representative
  nearest-neighbor query.
- Run `monitoring.sql` through a read-only monitoring integration. Alert on
  disk usage, connection saturation, replication/backup age, dead tuples,
  lock waits, slow vector queries, and HNSW recall/latency regressions.
- Leave autovacuum enabled. Run `ANALYZE mbzuai_retrieval.embedding_records`
  after a large release finishes loading, then compare `EXPLAIN (ANALYZE,
  BUFFERS)` and retrieval-quality results before tuning HNSW parameters.
- Patch by changing the pinned image deliberately, testing backup/restore and
  query recall in staging, then using a maintenance window. Never deploy a
  floating `latest` tag.
