"""Versioned document representation contracts."""

from pipeline.core.representation_v2 import (
    REPRESENTATION_V2_KIND,
    REPRESENTATION_V2_SCHEMA_VERSION,
    attach_page_links_to_documents,
    representation_v2_json_schema,
    stable_representation_id,
    validate_representation_v2,
)

__all__ = [
    "REPRESENTATION_V2_KIND",
    "REPRESENTATION_V2_SCHEMA_VERSION",
    "attach_page_links_to_documents",
    "representation_v2_json_schema",
    "stable_representation_id",
    "validate_representation_v2",
]
