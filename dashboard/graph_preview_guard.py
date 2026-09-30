"""Prevent graph-preview registrations from being used to mutate source runs."""

from starlette.responses import JSONResponse


def preview_guard(get_run_type):
    async def guard(request, call_next):
        parts = request.url.path.strip("/").split("/")
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and len(parts) >= 3
            and parts[:2] == ["api", "runs"]
            and parts[2].isdigit()
        ):
            # Removing a registration only deletes local metadata, not artifacts.
            deleting_registration = request.method == "DELETE" and len(parts) == 3
            if not deleting_registration and get_run_type(int(parts[2])) == "graph_preview":
                return JSONResponse(
                    {
                        "detail": "Graph previews are read-only. Pipeline execution and mutations are disabled."
                    },
                    status_code=403,
                )
        return await call_next(request)

    return guard
