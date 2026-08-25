-- Run with the admin role. This contains no application data or credentials.
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector', 'pg_stat_statements')
ORDER BY extname;

SELECT project_name, status, count(*) AS releases, max(updated_at) AS last_updated_at
FROM mbzuai_retrieval.releases
GROUP BY project_name, status
ORDER BY project_name, status;

SELECT release_id, lane, count(*) AS vectors
FROM mbzuai_retrieval.embedding_records
GROUP BY release_id, lane
ORDER BY release_id, lane;

SELECT calls, mean_exec_time, rows, left(query, 160) AS query
FROM pg_stat_statements
WHERE query LIKE '%mbzuai_retrieval.embedding_records%'
ORDER BY total_exec_time DESC
LIMIT 20;
