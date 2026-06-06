import fitz
import pandas as pd
import json
import markdown
import pytesseract

from pdf2image import convert_from_path
from bs4 import BeautifulSoup
from docx import Document
from PIL import Image


# =====================================
# IMAGE OCR
# =====================================

def extract_text_from_image(file_path):

    image = Image.open(file_path)

    text = pytesseract.image_to_string(

        image,

        config="--psm 6"
    )

    return text


# =====================================
# SCANNED PDF OCR
# =====================================

def extract_scanned_pdf_text(file_path):

    images = convert_from_path(file_path)

    pages = []

    for i, image in enumerate(images):

        page_text = pytesseract.image_to_string(

            image,

            config="--psm 6"
        )

        pages.append({

            "page": i + 1,

            "text": page_text
        })

    return pages


# =====================================
# PDF
# =====================================

def extract_text_from_pdf(file_path):

    doc = fitz.open(file_path)

    pages = []

    for page_num in range(len(doc)):

        page = doc[page_num]

        page_text = page.get_text()

        pages.append({

            "page": page_num + 1,

            "text": page_text
        })

    total_text = "".join(

        p["text"]

        for p in pages
    )

    if len(total_text.strip()) < 20:

        print(

            "Scanned PDF detected. Using OCR..."
        )

        pages = extract_scanned_pdf_text(
            file_path
        )

    return pages


# =====================================
# DOCX
# =====================================

def extract_text_from_docx(file_path):

    doc = Document(file_path)

    text = ""

    for para in doc.paragraphs:

        text += para.text + "\n"

    return text


# =====================================
# TXT
# =====================================

def extract_text_from_txt(file_path):

    with open(

        file_path,

        "r",

        encoding="utf-8"

    ) as f:

        return f.read()


# =====================================
# CSV
# =====================================

def extract_text_from_csv(file_path):

    df = pd.read_csv(file_path)

    return df.to_string()


# =====================================
# XLSX
# =====================================

def extract_text_from_xlsx(file_path):

    df = pd.read_excel(file_path)

    return df.to_string()


# =====================================
# JSON
# =====================================

def extract_text_from_json(file_path):

    with open(

        file_path,

        "r",

        encoding="utf-8"

    ) as f:

        data = json.load(f)

    return json.dumps(

        data,

        indent=2
    )


# =====================================
# MARKDOWN
# =====================================

def extract_text_from_md(file_path):

    with open(

        file_path,

        "r",

        encoding="utf-8"

    ) as f:

        md_text = f.read()

    html = markdown.markdown(
        md_text
    )

    soup = BeautifulSoup(

        html,

        "html.parser"
    )

    return soup.get_text()

