from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def read_and_clean_file(file_path):
    with open(file_path, 'r') as file:
        content = '\n'.join(line.strip() for line in file if line.strip())
    return content

def extract_invoice_data(file_path):
    system_prompt = """
    You are an expert question answering bot designed to search and extract answers from unstructured data in investment memo document.
   
    """
    extracted_text_from_pdf = read_and_clean_file(file_path)
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
                    {"type": "text", "text": f"Using this text from investment memo PPT page, answer the following questions. \n \
                     Text: {extracted_text_from_pdf}. \n\n\n Question: Who are the key executives of the company, including the CEO and CFO, and what are their relevant backgrounds and experiences?"},
                
                ]
            }
        ],
        temperature=0.0,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

if __name__ == "__main__":
    txt_path = './data/sonder_extracted.txt'
    extract_invoice_data(txt_path)
