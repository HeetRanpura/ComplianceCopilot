import io
import os

def normalize_text(text: str) -> str:
    """
    Improved text normalization for ingestion pipeline.
    Fixes whitespace, non-breaking spaces, and excessive gaps.
    """
    if not isinstance(text, str):
        return text
    text = text.replace("\xa0", " ")
    text = " ".join(text.split())  # collapse multiple whitespace
    return text.strip()


def extract_text_from_uploaded(uploaded_file):
    """Support .pdf, .txt, .docx. Returns best-effort extracted text."""
    name = getattr(uploaded_file, 'name', '').lower() if uploaded_file is not None else ''

    try:
        # ------------------------------
        # TXT extraction
        # ------------------------------
        if name.endswith('.txt'):
            raw = uploaded_file.getvalue().decode('utf-8', errors='replace')
            return normalize_text(raw)

        # ------------------------------
        # PDF extraction
        # ------------------------------
        if name.endswith('.pdf'):
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                out = []
                for p in reader.pages:
                    try:
                        out.append(p.extract_text() or "")
                    except Exception:
                        out.append("")
                raw = "\n\n".join(out)
                return normalize_text(raw)

            except Exception:
                # fallback: raw binary decode
                try:
                    raw = uploaded_file.getvalue().decode('utf-8', errors='replace')
                    return normalize_text(raw)
                except Exception:
                    return ""

        # ------------------------------
        # DOCX extraction
        # ------------------------------
        if name.endswith('.docx'):
            try:
                from docx import Document
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                doc = Document(tmp_path)
                paragraphs = [p.text for p in doc.paragraphs]

                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                raw = "\n\n".join(paragraphs)
                return normalize_text(raw)

            except Exception:
                # fallback: raw decode
                try:
                    raw = uploaded_file.getvalue().decode('utf-8', errors='replace')
                    return normalize_text(raw)
                except Exception:
                    return ""

        # ------------------------------
        # Default fallback
        # ------------------------------
        try:
            raw = uploaded_file.getvalue().decode('utf-8', errors='replace')
            return normalize_text(raw)
        except Exception:
            return ""

    except Exception:
        return ""
