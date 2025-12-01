import io
import os

def extract_text_from_uploaded(uploaded_file):
    """Support .pdf, .txt, .docx. Returns best-effort extracted text."""
    name = getattr(uploaded_file, 'name', '').lower() if uploaded_file is not None else ''
    try:
        if name.endswith('.txt'):
            text = uploaded_file.getvalue().decode('utf-8', errors='replace')
            return text

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
                return "\n\n".join(out).strip()
            except Exception:
                try:
                    data = uploaded_file.getvalue()
                    return data.decode('utf-8', errors='replace')
                except Exception:
                    return ""

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
                return "\n\n".join(paragraphs).strip()
            except Exception:
                try:
                    return uploaded_file.getvalue().decode('utf-8', errors='replace')
                except Exception:
                    return ""

        try:
            return uploaded_file.getvalue().decode('utf-8', errors='replace')
        except Exception:
            return ""
    except Exception:
        return ""
