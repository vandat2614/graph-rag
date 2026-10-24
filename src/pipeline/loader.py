import re
from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """Read all text from a PDF, joining pages."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
            pages.append(text.strip())
    return "\n".join(pages)
