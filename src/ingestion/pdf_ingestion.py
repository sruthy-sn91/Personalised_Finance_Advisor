import PyPDF2
import io

class PDFIngestion:
    def extract_text_from_pdf(self, pdf_path):
        text_content = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_content.append(page.extract_text())
        return "\n".join(text_content)
