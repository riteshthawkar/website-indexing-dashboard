"""Multi-source ingestion foundation, independent of MBZUAI deployment profiles."""

from .catalog import IngestionCatalog
from .config import MANIFEST_SCHEMA_VERSION, load_collection
from .contracts import (
    AccessContext, AccessPolicy, Collection, SnapshotEnd, Source, SourceConnector, SourceItem,
    SyncPolicy,
)
from .evaluation import build_evaluation_corpus
from .documents import ConversionFailure, ConversionOptions, DocumentProcessor, PreparedDocument
from ._prepared import PreparedRecord, read_prepared
from .folders import LocalFolderConnector
from .object_storage import S3Connector
from .preparation import PreparationReport, preparation_report, prepare_collection
from .permissions import update_source_access
from .records import ContentRecord, RECORD_SCHEMA_VERSION
from .sync import SyncError, SyncReport, sync_source
from .websites import WebsiteRunConnector


__all__ = [
    "AccessContext", "AccessPolicy", "Collection", "ContentRecord", "IngestionCatalog",
    "LocalFolderConnector", "MANIFEST_SCHEMA_VERSION", "RECORD_SCHEMA_VERSION", "SnapshotEnd",
    "Source", "SourceConnector", "SourceItem", "SyncError", "SyncPolicy", "SyncReport",
    "WebsiteRunConnector", "build_evaluation_corpus", "load_collection", "sync_source",
    "update_source_access",
    "S3Connector", "ConversionFailure", "ConversionOptions", "DocumentProcessor",
    "PreparedDocument", "PreparedRecord", "PreparationReport", "read_prepared",
    "prepare_collection", "preparation_report",
]
