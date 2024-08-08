from PyPDF2 import PdfReader
from py_pdf_parser.loaders import load_file
from pprint import pprint
import pdfplumber

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_information(self):
        with open(self.pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            information = pdf.getDocumentInfo()
            number_of_pages = pdf.getNumPages()

        txt = f"""
        Information about {self.pdf_path}: 

        Author: {information.author}
        Creator: {information.creator}
        Producer: {information.producer}
        Subject: {information.subject}
        Title: {information.title}
        Number of pages: {number_of_pages}
        """
        # print(txt)
        return information

    def extract_reference_elements(self):
        document = load_file(self.pdf_path)

        to_element = document.elements.filter_by_text_equal("TO:").extract_single_element()
        from_element = document.elements.filter_by_text_equal("FROM:").extract_single_element()
        date_element = document.elements.filter_by_text_equal("DATE:").extract_single_element()
        subject_element = document.elements.filter_by_text_equal("SUBJECT:").extract_single_element()

        to_text = document.elements.to_the_right_of(to_element).extract_single_element().text()
        from_text = document.elements.to_the_right_of(from_element).extract_single_element().text()
        date_text = document.elements.to_the_right_of(date_element).extract_single_element().text()
        subject_text_element = document.elements.to_the_right_of(subject_element).extract_single_element()
        subject_text = subject_text_element.text()

        content_elements = document.elements.after(subject_element)
        content_text = "\n".join(element.text() for element in content_elements)

        output = {
            "to": to_text,
            "from": from_text,
            "date": date_text,
            "subject": subject_text,
            "content": content_text,
        }

        # pprint(output)
        return output

    def extract_text_data(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            text = pdf.extract_text()
            print(text)

            tables = pdf.extract_table()
            for table in tables:
                print(table)

            images = pdf.get_images()
            for image in images:
                print(image["page_number"])
                with open(f"image_{image['page_number']}.jpg", "wb") as f:
                    f.write(image["data"])

        return text, tables, images

if __name__ == '__main__':
    path = './investor_decks/sonder.pdf'
    pdf_processor = PDFProcessor(path)
    pdf_processor.extract_information()
    pdf_processor.extract_reference_elements()
    pdf_processor.extract_text_data()
