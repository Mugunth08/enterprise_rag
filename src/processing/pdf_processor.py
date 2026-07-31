from pathlib import Path

import fitz
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# Load PDF
# -----------------------------
loader = PyPDFLoader("./data/raw/science.pdf")
documents = loader.load()

# -----------------------------
# Create Output Directories
# -----------------------------
text_dir = Path("./data/processed/text")
image_dir = Path("./data/processed/images")

text_dir.mkdir(parents=True, exist_ok=True)
image_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Save Text
# -----------------------------
for index, document in enumerate(documents, start=1):
    output_file = text_dir / f"page_{index}.txt"

    output_file.write_text(
        document.page_content,
        encoding="utf-8"
    )

print(f"Saved {len(documents)} text files.")

# -----------------------------
# Extract Images
# -----------------------------
pdf = fitz.open("./data/raw/science.pdf")

image_count = 0

for page_number, page in enumerate(pdf, start=1):

    images = page.get_images(full=True)

    for image_index, image in enumerate(images, start=1):

        xref = image[0]

        image_data = pdf.extract_image(xref)

        image_bytes = image_data["image"]
        image_extension = image_data["ext"]

        image_path = image_dir / f"page_{page_number}_image_{image_index}.{image_extension}"

        with open(image_path, "wb") as file:
            file.write(image_bytes)

        image_count += 1

pdf.close()

print(f"Saved {image_count} images.")