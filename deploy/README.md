# Retriever deployment contract

The default production image command runs
`scripts/deploy/start-retriever-from-active-release.sh`. Do not override it in
DigitalOcean: it hydrates or mounts the release, validates every immutable
artifact, and only then starts the retriever.

Two storage modes are supported:

- `persistent` is recommended for a Droplet with a mounted Volume. It requires
  `/data/releases/.mbzuai-release-storage` and refuses to start without it.
- `hydrate` is the App Platform fallback. It downloads a minimal runtime
  archive from an allowlisted HTTPS/Spaces host, verifies an externally
  configured SHA256, safely extracts it into ephemeral storage, and applies the
  same release checks.

App Platform has no persistent volumes and a limited ephemeral filesystem, so
the full crawler/indexer must run on a Droplet or build worker. Use the
lightweight `Dockerfile.retriever` for the App Platform service and keep runtime
archives below the configured compressed and extracted size limits.

The committed [`.do/app.yaml`](../.do/app.yaml) is the production structure for
one public backend plus one private retriever. It pins `Dockerfile.retriever`,
port 8060 as internal-only, 16 GB retriever memory, long hydration health-check
delays, release identity, and required mode. Never apply the checked-in
`CHANGE_ME` secret placeholders directly: copy the spec privately, populate
the DigitalOcean encrypted values, then update the existing app. The pre-deploy
suite validates the committed structure on every push.

Build an archive only from an already promoted release:

```bash
python scripts/deploy/build-runtime-release-archive.py \
  --active-release-file /opt/mbzuai/releases/mbzuai_main/active_release.json \
  --runs-root /opt/mbzuai/releases/runs/mbzuai_main \
  --output /tmp/mbzuai-runtime-release.tar.gz
```

The command writes a sibling `.sha256` file. Publish archives under immutable
object names containing the release ID; never replace an existing object.

After startup, run:

```bash
bash scripts/deploy/smoke-retriever.sh
```

The smoke verifies the live commit SHA, release/run identity, and bundle hash
against the locally validated active pointer. When running it remotely, supply
`EXPECTED_RETRIEVER_COMMIT_SHA`, `EXPECTED_RETRIEVAL_RUN_ID`,
`EXPECTED_RETRIEVAL_BUNDLE_SHA256`, and optionally
`EXPECTED_RETRIEVAL_RELEASE_ID`. It also requires the same
`RETRIEVAL_SERVICE_TOKEN` used by the backend.

Persistent rollback is atomic and revalidates the target:

```bash
ROLLBACK_RUN_ID=<previous-run-id> \
bash scripts/deploy/rollback-active-release.sh
```

Hydrated rollback points `RELEASE_ARCHIVE_S3_URI` and `RELEASE_ARCHIVE_SHA256`
at the previous immutable object, then performs a fresh deployment. The
retriever authenticates to the exact allowlisted HTTPS endpoint with a
read-only object-store key; no presigned URL is required. Copy
`deploy/production.env.example` into the deployment control plane and never
commit populated credentials.

Automatic deploy-on-push is intentionally disabled for both components. A push
must first pass the backend and indexing pre-deploy workflows. Then build and
promote against the exact commits, publish the immutable runtime archive, set
its S3 URI/endpoint/region/SHA/host and the expected run/bundle identity in DigitalOcean, and
initiate the production deployment from the protected release process. A
mismatched old archive is rejected by the serving fingerprint. After rotating
Pinecone credentials, update the private retriever component's encrypted
`PINECONE_API_KEY` (not the public backend component); the mandatory startup
query must reach the live provider before `/readyz` succeeds.

Each retriever process deliberately serves one retrieval at a time because the
current routed retriever is not declared safe for shared parallel calls. The
10-second bounded queue absorbs short bursts. Add capacity with horizontally
isolated retriever replicas only after the multi-replica soak gate passes; do
not increase `RETRIEVER_MAX_CONCURRENCY` within one process.
