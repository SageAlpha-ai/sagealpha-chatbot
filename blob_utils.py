"""
SageAlpha.ai Azure Blob Storage Utilities
Handles blob operations for file storage and retrieval
"""

from typing import List

import requests
from azure.storage.blob import BlobProperties, BlobServiceClient


class BlobReader:
    """Azure Blob Storage reader for downloading and listing blobs."""

    def __init__(self, conn_str: str, container: str) -> None:
        """
        Initialize BlobReader with Azure connection string.

        Args:
            conn_str: Azure Storage connection string
            container: Container name to operate on
        """
        self.conn_str = conn_str
        self.container = container
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container_client = self.client.get_container_client(container)

    def list_blobs(self, prefix: str = "") -> List[BlobProperties]:
        """
        List blobs in the container with optional prefix filter.

        Args:
            prefix: Optional prefix to filter blobs

        Returns:
            List of BlobProperties objects
        """
        return list(self.container_client.list_blobs(name_starts_with=prefix))

    def download_blob_to_bytes(self, blob_path: str) -> bytes:
        """
        Download a blob to bytes.

        Args:
            blob_path: Path to the blob within the container

        Returns:
            Blob content as bytes
        """
        blob_client = self.client.get_blob_client(
            container=self.container, blob=blob_path
        )
        stream = blob_client.download_blob()
        return stream.readall()

    def download_blob_url_to_bytes(self, url: str, timeout: int = 30) -> bytes:
        """
        Download content from a URL (typically a blob URL with SAS token).

        Args:
            url: Full URL to download from
            timeout: Request timeout in seconds

        Returns:
            Downloaded content as bytes
        """
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content

    def get_blob_properties(self, blob_path: str) -> BlobProperties:
        """
        Get properties of a specific blob.

        Args:
            blob_path: Path to the blob within the container

        Returns:
            BlobProperties object
        """
        blob_client = self.client.get_blob_client(
            container=self.container, blob=blob_path
        )
        return blob_client.get_blob_properties()

    def blob_exists(self, blob_path: str) -> bool:
        """
        Check if a blob exists.

        Args:
            blob_path: Path to the blob within the container

        Returns:
            True if blob exists, False otherwise
        """
        blob_client = self.client.get_blob_client(
            container=self.container, blob=blob_path
        )
        return blob_client.exists()
