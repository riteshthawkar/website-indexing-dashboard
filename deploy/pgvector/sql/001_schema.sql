\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

DO $roles$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mbzuai_retrieval_reader') THEN
        CREATE ROLE mbzuai_retrieval_reader NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mbzuai_retrieval_writer') THEN
        CREATE ROLE mbzuai_retrieval_writer NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
END
$roles$;

CREATE SCHEMA IF NOT EXISTS mbzuai_retrieval;
REVOKE ALL ON SCHEMA mbzuai_retrieval FROM PUBLIC;

CREATE TABLE IF NOT EXISTS mbzuai_retrieval.schema_migrations (
    version integer PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mbzuai_retrieval.releases (
    release_id text PRIMARY KEY,
    project_name text NOT NULL,
    status text NOT NULL DEFAULT 'building',
    embedding_model text NOT NULL,
    embedding_dimensions integer NOT NULL,
    distance_metric text NOT NULL DEFAULT 'cosine',
    contract_sha256 text NOT NULL,
    expected_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
    artifact_hashes jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    ready_at timestamptz,
    activated_at timestamptz,
    retired_at timestamptz,
    CONSTRAINT releases_status_check
        CHECK (status IN ('building', 'ready', 'active', 'retired', 'failed')),
    CONSTRAINT releases_dimensions_check CHECK (embedding_dimensions = 1536),
    CONSTRAINT releases_metric_check CHECK (distance_metric = 'cosine'),
    CONSTRAINT releases_contract_sha_check CHECK (contract_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT releases_expected_counts_object_check CHECK (jsonb_typeof(expected_counts) = 'object'),
    CONSTRAINT releases_artifact_hashes_object_check CHECK (jsonb_typeof(artifact_hashes) = 'object')
);

CREATE UNIQUE INDEX IF NOT EXISTS releases_one_active_project_idx
    ON mbzuai_retrieval.releases (project_name)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS releases_project_status_idx
    ON mbzuai_retrieval.releases (project_name, status, created_at DESC);

CREATE TABLE IF NOT EXISTS mbzuai_retrieval.embedding_records (
    row_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    release_id text NOT NULL
        REFERENCES mbzuai_retrieval.releases (release_id) ON DELETE RESTRICT,
    namespace text NOT NULL,
    lane text NOT NULL,
    record_id text NOT NULL,
    retrieval_text text NOT NULL,
    source_url text,
    language text,
    content_sha256 text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536) NOT NULL,
    search_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(retrieval_text, ''))
    ) STORED,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT embedding_records_lane_check CHECK (
        lane IN (
            'chunks', 'parents', 'media', 'facts', 'evidence_spans',
            'summaries', 'assertions', 'entities', 'communities'
        )
    ),
    CONSTRAINT embedding_records_namespace_check CHECK (length(namespace) BETWEEN 1 AND 255),
    CONSTRAINT embedding_records_record_id_check CHECK (length(record_id) BETWEEN 1 AND 1024),
    CONSTRAINT embedding_records_text_check CHECK (length(retrieval_text) BETWEEN 1 AND 1000000),
    CONSTRAINT embedding_records_content_sha_check CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT embedding_records_metadata_object_check CHECK (jsonb_typeof(metadata) = 'object'),
    CONSTRAINT embedding_records_release_lane_id_unique UNIQUE (release_id, lane, record_id)
);

CREATE INDEX IF NOT EXISTS embedding_records_release_namespace_idx
    ON mbzuai_retrieval.embedding_records (release_id, namespace);
CREATE INDEX IF NOT EXISTS embedding_records_release_lane_idx
    ON mbzuai_retrieval.embedding_records (release_id, lane, record_id);
CREATE INDEX IF NOT EXISTS embedding_records_source_url_idx
    ON mbzuai_retrieval.embedding_records (release_id, source_url)
    WHERE source_url IS NOT NULL;
CREATE INDEX IF NOT EXISTS embedding_records_language_idx
    ON mbzuai_retrieval.embedding_records (release_id, language)
    WHERE language IS NOT NULL;
CREATE INDEX IF NOT EXISTS embedding_records_metadata_gin_idx
    ON mbzuai_retrieval.embedding_records USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS embedding_records_search_tsv_idx
    ON mbzuai_retrieval.embedding_records USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS embedding_records_embedding_hnsw_idx
    ON mbzuai_retrieval.embedding_records
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

INSERT INTO mbzuai_retrieval.schema_migrations (version)
VALUES (1)
ON CONFLICT (version) DO NOTHING;

REVOKE ALL ON ALL TABLES IN SCHEMA mbzuai_retrieval FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA mbzuai_retrieval FROM PUBLIC;

GRANT USAGE ON SCHEMA mbzuai_retrieval TO mbzuai_retrieval_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA mbzuai_retrieval TO mbzuai_retrieval_reader;

GRANT mbzuai_retrieval_reader TO mbzuai_retrieval_writer;
GRANT INSERT, UPDATE ON mbzuai_retrieval.releases TO mbzuai_retrieval_writer;
GRANT INSERT, UPDATE ON mbzuai_retrieval.embedding_records TO mbzuai_retrieval_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA mbzuai_retrieval TO mbzuai_retrieval_writer;

ALTER DEFAULT PRIVILEGES IN SCHEMA mbzuai_retrieval
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA mbzuai_retrieval
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA mbzuai_retrieval
    GRANT SELECT ON TABLES TO mbzuai_retrieval_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA mbzuai_retrieval
    GRANT INSERT, UPDATE ON TABLES TO mbzuai_retrieval_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA mbzuai_retrieval
    GRANT USAGE, SELECT ON SEQUENCES TO mbzuai_retrieval_writer;
