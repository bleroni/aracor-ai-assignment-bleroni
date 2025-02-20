import os
import pdfplumber
from docx import Document
from docx.opc.exceptions import PackageNotFoundError


class UnsupportedFormatError(Exception):
    """Raised when an unsupported file format is encountered"""
    pass


class CorruptedFileError(Exception):
    """Raised when a file is corrupted or cannot be processed"""
    pass


class DocumentProcessor:
    SUPPORTED_FORMATS = ['pdf', 'txt', 'docx']

    def __init__(self):
        self.base_dir = "src/documents"

    def process_file(self, file_path):
        """
        Process a document file and extract its text content
        Args:
            file_path (str): Path to the document file
        Returns:
            str: Extracted text content
        Raises:
            UnsupportedFormatError: For unsupported file formats
            CorruptedFileError: For corrupted or unreadable files
        """
        file_extension = self._get_file_extension(file_path)

        if file_extension not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(f"Unsupported file format: {file_extension}") # noqa

        try:
            if file_extension == "pdf":
                return self._extract_pdf_text(file_path)
            elif file_extension == "txt":
                return self._extract_txt_text(file_path)
            elif file_extension == "docx":
                return self._extract_docx_text(file_path)
        except Exception as e:
            raise CorruptedFileError(f"Failed to process file {file_path}: {str(e)}") from e

    def _get_file_extension(self, file_path):
        """Extract and normalize file extension"""
        _, ext = os.path.splitext(file_path)
        return ext.lower().lstrip('.')

    def _extract_pdf_text(self, file_path):
        """Extract text from PDF using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ''
            return text
        except Exception as e:
            raise CorruptedFileError(f"PDF processing error: {str(e)}") from e

    def _extract_txt_text(self, file_path):
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise CorruptedFileError(f"Text file processing error: {str(e)}") from e

    def _extract_docx_text(self, file_path):
        """Extract text from DOCX using python-docx"""
        try:
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except PackageNotFoundError as e:
            raise CorruptedFileError("Invalid or corrupted DOCX file") from e
        except Exception as e:
            raise CorruptedFileError(f"DOCX processing error: {str(e)}") from e


if __name__ == "__main__":
    processor = DocumentProcessor()
    files = os.listdir(processor.base_dir)

    for file in files:
        file_path = f"{processor.base_dir}/{file}"
        try:
            text = processor.process_file(file_path)
            print(f"Processed {file} successfully. Text length: {len(text)} characters.")
        except UnsupportedFormatError as e:
            print(f"Unsupported format: {e}")
        except CorruptedFileError as e:
            print(f"File error: {e}")
