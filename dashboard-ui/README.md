# Dashboard UI

This is the React / Next.js frontend for the website indexing dashboard.

It talks to the FastAPI backend in [`dashboard/`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard).

## Start In Dev Mode

```bash
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Open:

- `http://127.0.0.1:3000`

The UI automatically points to `http://127.0.0.1:8050` when served locally on `:3000`, unless `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_WS_URL` are set.

## Production-Like Local Serve

This app uses static export output:

- [`next.config.ts`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/next.config.ts)

Build and serve:

```bash
npm run build
npx serve out -l 3000
```

## Root Product Docs

See the root setup docs for full machine setup:

- [`README.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/README.md)
- [`docs/setup.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/setup.md)
