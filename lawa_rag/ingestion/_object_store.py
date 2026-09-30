"""Read-only, bounded S3 listing and conditional download helpers."""

from __future__ import annotations

from pathlib import PurePosixPath


def inventory(client, bucket: str, prefix: str, extensions: tuple[str, ...],
              limit: int, max_pages: int, max_bytes: int) -> dict:
    result, tokens, total_bytes = {}, set(), 0
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": 1000},
    )
    for page_number, response in enumerate(pages, 1):
        if not isinstance(response.get("IsTruncated"), bool):
            raise ValueError("Object listing did not confirm whether it is complete")
        for value in response.get("Contents", []):
            key = value.get("Key")
            if not isinstance(key, str) or not key.startswith(prefix):
                raise ValueError("Object listing returned an out-of-scope key")
            path = PurePosixPath(key)
            if key.endswith("/") or any(part.startswith(".") for part in path.parts):
                continue
            if path.suffix.lower() not in extensions:
                continue
            if key in result:
                raise ValueError("Object listing repeated a key")
            result[key] = object_identity(value)
            total_bytes += result[key][1]
            if len(result) > limit:
                raise ValueError("Object limit exceeded; listing was not truncated")
            if total_bytes > max_bytes:
                raise ValueError("Snapshot exceeds its total byte budget")
        if not response["IsTruncated"]:
            return result
        if page_number == max_pages:
            raise ValueError("Listing page limit exceeded; snapshot is incomplete")
        token = response.get("NextContinuationToken")
        if not isinstance(token, str) or not token or token in tokens:
            raise ValueError("Truncated listing has no new continuation token")
        tokens.add(token)
    raise ValueError("Object listing ended without confirming completeness")


def object_identity(value: dict) -> tuple[str, int, str]:
    etag, size, modified = value.get("ETag"), value.get("Size"), value.get("LastModified")
    if not isinstance(etag, str) or not etag or not hasattr(modified, "isoformat"):
        raise ValueError("Object listing has incomplete revision metadata")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("Object listing has an invalid size")
    return etag, size, modified.isoformat()


def download(client, bucket: str, key: str, identity: tuple, max_bytes: int) -> bytes:
    etag, size, modified = identity
    if size > max_bytes:
        raise ValueError("Object exceeds the configured size limit")
    response = client.get_object(Bucket=bucket, Key=key, IfMatch=etag)
    stream = response.get("Body")
    if stream is None:
        raise ValueError("Object response has no body")
    try:
        actual = object_identity({**response, "Size": response.get("ContentLength")})
        if actual != (etag, size, modified):
            raise ValueError("Object changed between listing and download")
        body = stream.read(max_bytes + 1)
        if len(body) != size:
            raise ValueError("Object body is truncated or exceeds its declared size")
        return body
    finally:
        stream.close()
