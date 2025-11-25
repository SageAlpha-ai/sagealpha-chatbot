import os
import time
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from . import NseUtility  # Assuming this imports the file containing NseUtils class
from io import StringIO, BytesIO
from datetime import datetime

# 1. Load Environment Variables
load_dotenv()

CONNECTION_STRING = os.getenv("AZURE_CONN_STR")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "nse-data-raw")

# Keywords to filter specifically for Financial Reports
FINANCIAL_KEYWORDS = ["Financial Result", "Outcome of Board Meeting", "Audited", "Unaudited"]

def get_blob_service_client():
    """Creates the Azure Client securely"""
    if not CONNECTION_STRING:
        raise ValueError("AZURE_CONN_STR is missing in .env file!")
    return BlobServiceClient.from_connection_string(CONNECTION_STRING)

def upload_csv_to_azure(df, blob_name, client):
    """
    Uploads the list of announcements (Metadata) as a CSV file.
    Sets Content-Type to 'text/csv' so it opens in Excel/Preview.
    """
    try:
        blob_client = client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        # Convert DataFrame to CSV in memory
        output = StringIO()
        df.to_csv(output, index=False)
        client_data = output.getvalue()
        
        print(f"Uploading Metadata: {blob_name}...")
        
        # Upload with specific content settings (The FIX)
        blob_client.upload_blob(
            client_data, 
            overwrite=True, 
            content_settings=ContentSettings(content_type='text/csv')
        )
        print(f" -> Success: Metadata uploaded.")
        
    except Exception as e:
        print(f"Failed to upload CSV {blob_name}: {e}")

def process_documents(df, nse, client):
    """
    Iterates through the DataFrame.
    If it finds a Financial Report, it downloads the PDF and uploads to Azure.
    """
    print(f"Scanning {len(df)} announcements for Financial Reports...")
    
    container_client = client.get_container_client(CONTAINER_NAME)
    downloaded_count = 0

    for index, row in df.iterrows():
        subject = str(row.get('desc', '')).lower()
        attachment = row.get('attchmntFile')  # FIXED: Use correct column name from API
        symbol = row.get('symbol')
        
        # Check if it's a financial keyword AND has a file
        if attachment and any(k.lower() in subject for k in FINANCIAL_KEYWORDS):
            
            # define blob path: Symbol / Year / Filename (extract filename from URL)
            filename = os.path.basename(attachment)
            blob_path = f"documents/{symbol}/{datetime.now().year}/{filename}"
            
            # Check if exists to avoid re-downloading (Optimization)
            blob_client = container_client.get_blob_client(blob_path)
            
            if not blob_client.exists():
                print(f" -> Downloading PDF for {symbol}: {filename}")
                
                # Download using full URL
                pdf_content = nse.download_document(attachment)
                
                if pdf_content:
                    # Upload the PDF
                    blob_client.upload_blob(
                        pdf_content, 
                        overwrite=True,
                        content_settings=ContentSettings(content_type='application/pdf')
                    )
                    print(f"    [Uploaded] {blob_path}")
                    downloaded_count += 1
                else:
                    print(f"    [Failed] Could not download from NSE")
                    time.sleep(2)  # Brief pause on failure to avoid rate limits
            else:
                print(f"    [Skipped] Already exists: {blob_path}")

    print(f"Processed: {downloaded_count} new PDFs uploaded.")

def main():
    # Initialize Azure
    try:
        blob_service = get_blob_service_client()
        # Create container if not exists
        if not blob_service.get_container_client(CONTAINER_NAME).exists():
            blob_service.create_container(CONTAINER_NAME)
            print(f"Created container: {CONTAINER_NAME}")
    except Exception as e:
        print(f"Azure Connection Failed: {e}")
        return

    # Initialize NSE
    print("Initializing NSE Session...")
    try:
        nse = NseUtility.NseUtils()
    except Exception as e:
        print(f"Failed to initialize NSE Utility: {e}")
        return

    # Fetch Data
    print("Fetching Corporate Announcements...")
    df_announcements = nse.get_corporate_announcement()

    if df_announcements is not None and not df_announcements.empty:
        print(f"Fetched {len(df_announcements)} announcements.")
        
        # 1. Upload the Master CSV List
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"metadata/announcements_{timestamp}.csv"
        upload_csv_to_azure(df_announcements, csv_filename, blob_service)
        
        # 2. Download and Upload the specific PDF documents
        process_documents(df_announcements, nse, blob_service)
        
    else:
        print("No data fetched from NSE.")

if __name__ == "__main__":
    main()