"""Versioned read-only REST/OpenAPI surface."""

from .rest import API_PREFIX, ApiResponse, RestApplication, make_handler, openapi_document, serve

__all__ = ["API_PREFIX", "ApiResponse", "RestApplication", "make_handler",
           "openapi_document", "serve"]
