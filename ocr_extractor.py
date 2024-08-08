from transformers import LayoutLMv2Model, TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path

class PDFToTextConverter:
    def __init__(self, pdf_path, layoutlmv_model_name="microsoft/layoutxlm-base", trocr_processor_name="microsoft/trocr-base-handwritten", trocr_model_name="microsoft/trocr-base-handwritten"):
        self.pdf_path = pdf_path
        self.llmxlv_model = LayoutLMv2Model.from_pretrained(layoutlmv_model_name)
        self.processor = TrOCRProcessor.from_pretrained(trocr_processor_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name)

    def convert_pdf_to_images(self):
        """
        Convert PDF to images.
        """
        images = convert_from_path(self.pdf_path)
        return images

    def process_image(self, image):
        """
        Process an image to generate text.
        """
        image = image.convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def extract_text_from_pdf(self):
        """
        Extract text from each page of the PDF.
        """
        images = self.convert_pdf_to_images()
        text_output = []
        for i, image in enumerate(images):
            generated_text = self.process_image(image)
            text_output.append(f"Page {i+1} text: {generated_text}")
        return text_output

pdf_converter = PDFToTextConverter(pdf_path="path_to_your_pdf.pdf")
text_output = pdf_converter.extract_text_from_pdf()

for page_text in text_output:
    print(page_text)
