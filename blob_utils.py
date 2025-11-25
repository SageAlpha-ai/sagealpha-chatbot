import io
from azure.storage.blob import BlobServiceClient
import requests


class BlobReader:
    def __init__(self, conn_str: str, container: str):
        self.conn_str = conn_str
        self.container = container
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container_client = self.client.get_container_client(container)

    def list_blobs(self, prefix: str = ""):
        return list(self.container_client.list_blobs(name_starts_with=prefix))

    def download_blob_to_bytes(self, blob_path: str) -> bytes:
        blob_client = self.client.get_blob_client(
            container=self.container, blob=blob_path
        )
        stream = blob_client.download_blob()
        return stream.readall()

    def download_blob_url_to_bytes(self, url: str) -> bytes:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content

    def get_blob_properties(self, blob_path: str):
        blob_client = self.client.get_blob_client(
            container=self.container, blob=blob_path
        )
        return blob_client.get_blob_properties()
