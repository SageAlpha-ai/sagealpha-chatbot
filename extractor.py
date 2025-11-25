import io
from PyPDF2 import PdfReader
import xml.etree.ElementTree as ET
import re

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
            texts.append(txt)
        except Exception:
            continue
    return "\n".join(texts)

def parse_xbrl_file_to_text(xml_bytes: bytes) -> str:
    """
    Very small XBRL summarizer — extracts numeric facts and contexts.
    This is a simple extract; for production use consider Arelle or a robust XBRL parser.
    """
    root = ET.fromstring(xml_bytes)
    # Namespaces handling
    ns_map = {}
    for k, v in root.attrib.items():
        if k.startswith("xmlns"):
            parts = k.split(':')
            prefix = parts[1] if len(parts) > 1 else ''
            ns_map[prefix] = v

    facts = []
    # Heuristic: numeric elements often have 'decimals' or are in specific namespaces
    for elem in root.iter():
        tag = elem.tag
        if '}' in tag:
            tag_clean = tag.split('}', 1)[1]
        else:
            tag_clean = tag
        text = (elem.text or "").strip()
        if text:
            # simple numeric check
            if re.match(r"^[\d\.\-\,]+$", text) and len(text) < 30:
                facts.append(f"{tag_clean}: {text}")
            # also capture labeled items (like 'EntityRegistrantName' etc)
            if tag_clean.lower() in ('entityregistrantname', 'contextref', 'period', 'instant', 'startdate', 'enddate'):
                facts.append(f"{tag_clean}: {text}")
    # Fallback: full text few hundred chars
    summary = "\n".join(facts)
    if not summary:
        summary = (ET.tostring(root, encoding="unicode")[:2000])  # short fallback
    return summary