import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=100):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def extract_and_chunk(pdf_path, chunk_size=100):
    full_text = extract_text_from_pdf(pdf_path)
    return chunk_text(full_text, chunk_size)
