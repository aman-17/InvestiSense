from openai import OpenAI
import fitz
import io
import os
from PIL import Image
from dotenv import load_dotenv
import base64

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

@staticmethod
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pdf_to_base64_images(pdf_path):
    pdf_document = fitz.open(pdf_path)
    base64_images = []
    temp_image_paths = []

    total_pages = len(pdf_document)

    for page_num in range(total_pages):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        temp_image_path = f"temp_page_{page_num}.png"
        img.save(temp_image_path, format="PNG")
        temp_image_paths.append(temp_image_path)
        base64_image = encode_image(temp_image_path)
        base64_images.append(base64_image)

    for temp_image_path in temp_image_paths:
        os.remove(temp_image_path)

    return base64_images

def extract_invoice_data(base64_image):
    system_prompt = """
    You are an OCR-like data extraction tool designed to extract structured data from investment memo PPTs.
    1. Extraction Process: Extract data from the document, organizing it by theme or subgroup.
    2. Output Format: Present the extracted information in a structured text format, not JSON.
    3. Data Types: Extract various types of data, including but not limited to: Name of the company, Headquarter location, Various tables and charts, Business growth metrics, Revenue figures, Key customers, Team members.
    4. Empty Pages: If a page contains no relevant data, output "No relevant information on this page."
    5. Table Extraction: For tables, represent them in a clear, readable text format.
    6. Accuracy and Integrity: Do not interpolate or fabricate data under any circumstances.
    7. Maintain structural integrity of the information while presenting it in a readable, text-based format.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the data in this investment memo PPT page and output in a structured text format"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        temperature=0.0,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def extract_from_single_pdf(pdf_path, output_directory):
    base64_images = pdf_to_base64_images(pdf_path)
    entire_document = []

    for page_num, base64_image in enumerate(base64_images, start=1):
        extracted_text = extract_invoice_data(base64_image)
        entire_document.append(f"--- Page {page_num} ---\n{extracted_text}\n")

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct the output file path
    output_filename = os.path.join(output_directory, os.path.basename(pdf_path).replace('.pdf', '_extracted.txt'))
    
    # Save the entire_document as a text file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(entire_document))
    
    return output_filename

def main_extract(file_path, output_directory):
    if not os.path.isfile(file_path):
        print(f"Error: {file_path} is not a valid file.")
        return
    
    if not file_path.lower().endswith('.pdf'):
        print(f"Error: {file_path} is not a PDF file.")
        return
    output_filename = extract_from_single_pdf(file_path, output_directory)
    print(f"Extraction complete. Output saved to: {output_filename}")

if __name__ == "__main__":
    pdf_path = "./investor_decks/sonder.pdf"
    output_directory = "./data"
    main_extract(pdf_path, output_directory)
