import pytest
from docx import Document as DocxDocument

from src.processors.document_processor import CorruptedFileError, DocumentProcessor, UnsupportedFormatError


# 1. Test successful PDF processing
def test_process_pdf_success(tmp_path):
    # Create a temporary PDF file with a minimal valid structure that includes "Hello, PDF!"
    pdf_file = tmp_path / "test.pdf"
    pdf_content = b"""%PDF-1.1
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 55 >>
stream
BT
/F1 24 Tf
100 700 Td
(Hello, PDF!) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000111 00000 n 
0000000178 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
260
%%EOF
"""
    pdf_file.write_bytes(pdf_content)
    processor = DocumentProcessor()
    text = processor.process_file(str(pdf_file))
    # Check that the extracted text contains the expected string.
    assert "Hello, PDF!" in text


# 2. Test successful DOCX processing
def test_process_docx_success(tmp_path):
    # Create a temporary DOCX file using python-docx
    docx_file = tmp_path / "test.docx"
    doc = DocxDocument()
    doc.add_paragraph("Hello, DOCX!")
    doc.save(str(docx_file))

    processor = DocumentProcessor()
    text = processor.process_file(str(docx_file))
    assert "Hello, DOCX!" in text


# 3. Test successful TXT processing
def test_process_txt_success(tmp_path):
    # Create a temporary TXT file with known content
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello, TXT!", encoding="utf-8")

    processor = DocumentProcessor()
    text = processor.process_file(str(txt_file))
    assert text == "Hello, TXT!"


# 4. Test handling of corrupted files (using a corrupted PDF as an example)
def test_corrupted_file(tmp_path):
    # Create a temporary file with supported .pdf extension but invalid content
    corrupted_file = tmp_path / "corrupted.pdf"
    corrupted_file.write_bytes(b"This is not a valid PDF file content")

    processor = DocumentProcessor()
    with pytest.raises(CorruptedFileError):
        processor.process_file(str(corrupted_file))


# 5. Test handling of unsupported file formats
def test_unsupported_file_format(tmp_path):
    # Create a temporary file with an unsupported extension (e.g., .csv)
    unsupported_file = tmp_path / "test.csv"
    unsupported_file.write_text("some,data,here", encoding="utf-8")

    processor = DocumentProcessor()
    with pytest.raises(UnsupportedFormatError):
        processor.process_file(str(unsupported_file))


# 6. Test handling of empty files (using an empty TXT file as an example)
def test_empty_txt_file(tmp_path):
    # Create a temporary empty TXT file
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("", encoding="utf-8")

    processor = DocumentProcessor()
    text = processor.process_file(str(empty_txt))
    # For TXT files, reading an empty file should return an empty string
    assert text == ""
