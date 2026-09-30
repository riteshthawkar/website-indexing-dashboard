"""Real Boto3 client with its official HTTP-free Stubber boundary."""

from datetime import datetime, timezone
from io import BytesIO

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber

from lawa_rag.ingestion import AccessPolicy, S3Connector, Source


def stubbed_connector(**options):
    client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test-only",
                          aws_secret_access_key="test-only", endpoint_url="https://s3.example")
    source = Source("bucket", "s3", "s3://test-bucket/docs/", AccessPolicy(public=True))
    connector = S3Connector(source, client=client, **options)
    return connector, Stubber(client)


def objects(**values):
    return [{"Key": "docs/" + key, "Size": len(body), "ETag": '"opaque-' + key + '"',
             "LastModified": datetime(2026, 9, 17, tzinfo=timezone.utc)}
            for key, body in values.items()]


def listing(stubber, items, *, token=None, next_token=None, truncated=False):
    expected = {"Bucket": "test-bucket", "Prefix": "docs/", "MaxKeys": 1000}
    response = {"IsTruncated": truncated, "Contents": items}
    if token:
        expected["ContinuationToken"] = token
    if next_token:
        response["NextContinuationToken"] = next_token
    stubber.add_response("list_objects_v2", response, expected)


def object_body(stubber, item, body):
    stream = BytesIO(body)
    response = {"Body": StreamingBody(stream, len(body)), "ContentLength": len(body),
                "ETag": item["ETag"], "LastModified": item["LastModified"]}
    stubber.add_response("get_object", response, {"Bucket": "test-bucket", "Key": item["Key"],
                                                "IfMatch": item["ETag"]})
    return stream


def snapshot(stubber, values):
    items = objects(**values)
    listing(stubber, items)
    for item in sorted(items, key=lambda value: value["Key"]):
        object_body(stubber, item, values[item["Key"].removeprefix("docs/")])
    listing(stubber, items)
