\set ON_ERROR_STOP on

BEGIN;

-- Serialize operational migration attempts without holding a session-level
-- lock.  The lock is released automatically with this short transaction.
SELECT pg_advisory_xact_lock(hashtextextended('mbzuai_retrieval.schema_migration', 0));

ALTER TABLE mbzuai_retrieval.embedding_records
    DROP CONSTRAINT IF EXISTS embedding_records_lane_check;

ALTER TABLE mbzuai_retrieval.embedding_records
    ADD CONSTRAINT embedding_records_lane_check CHECK (
        lane IN (
            'chunks', 'parents', 'media', 'page_cards', 'actions',
            'facts', 'evidence_spans', 'summaries', 'assertions',
            'entities', 'communities'
        )
    ) NOT VALID;

ALTER TABLE mbzuai_retrieval.embedding_records
    VALIDATE CONSTRAINT embedding_records_lane_check;

INSERT INTO mbzuai_retrieval.schema_migrations (version)
VALUES (2)
ON CONFLICT (version) DO NOTHING;

COMMIT;
