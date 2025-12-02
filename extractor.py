"""
SageAlpha.ai Document Extraction Utilities
Extract text from PDF and XBRL files for indexing
"""

import io
import re
import xml.etree.ElementTree as ET
from typing import List

from PyPDF2 import PdfReader


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text content from PDF bytes.

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        Extracted text content as string
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts: List[str] = []

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
            if text.strip():
                texts.append(text)
        except Exception:
            continue

    return "\n".join(texts)


def extract_text_from_pdf_file(file_path: str) -> str:
    """
    Extract text content from a PDF file path.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content as string
    """
    with open(file_path, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())


def parse_xbrl_file_to_text(xml_bytes: bytes) -> str:
    """
    Parse XBRL/XML file and extract structured text.

    This is a simple extractor that captures numeric facts and key elements.
    For production use, consider using Arelle or a robust XBRL parser.

    Args:
        xml_bytes: Raw XML/XBRL file bytes

    Returns:
        Extracted text summary
    """
    root = ET.fromstring(xml_bytes)

    # Build namespace map for reference
    ns_map: dict = {}
    for key, value in root.attrib.items():
        if key.startswith("xmlns"):
            parts = key.split(":")
            prefix = parts[1] if len(parts) > 1 else ""
            ns_map[prefix] = value

    facts: List[str] = []

    # Important element tags to capture
    important_tags = {
        "entityregistrantname",
        "contextref",
        "period",
        "instant",
        "startdate",
        "enddate",
        "documenttype",
        "tradingsymbol",
        "securityexchangename",
    }

    for elem in root.iter():
        tag = elem.tag
        if "}" in tag:
            tag_clean = tag.split("}", 1)[1]
        else:
            tag_clean = tag

        text = (elem.text or "").strip()
        if not text:
            continue

        # Capture numeric values
        if re.match(r"^[\d\.\-\,]+$", text) and len(text) < 30:
            facts.append(f"{tag_clean}: {text}")

        # Capture important labeled items
        if tag_clean.lower() in important_tags:
            facts.append(f"{tag_clean}: {text}")

    # Build summary
    summary = "\n".join(facts)

    # Fallback if no facts extracted
    if not summary:
        summary = ET.tostring(root, encoding="unicode")[:2000]

    return summary


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excess whitespace and normalizing.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    return text.strip()
