from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import fitz  # PyMuPDF
from PIL import Image
import os
from tqdm import tqdm

# Load model and processor
model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use FP16 for better GPU performance
    _attn_implementation="eager"
).to("cuda")

pdf_path = "C:/Users/Rudy/Desktop/UN-run/undrr/action_plan.pdf"
output_txt_path = os.path.splitext(pdf_path)[0] + ".txt"

def pdf_to_images(pdf_path):
    """Converts each page of a PDF to an image."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)  # Adjust DPI for better speed/quality balance
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.resize((pix.width // 8, pix.height // 8))  # Reduce image size to speed up processing
        images.append(img)
    return images

def extract_text_from_image(image, prompt):
    """Processes a single image and extracts text using the model."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Pass PIL image correctly
                {"type": "text", "text": prompt},
            ]
        },
    ]

    with torch.no_grad():  # Disable gradients for faster inference
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.float16)

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def main():
    images = pdf_to_images(pdf_path)
    all_text = ""
    prompt = "Parse just the text and tables in this image into a dictionary format if table or just plain text if not. Do not summarize."

    for i, img in enumerate(tqdm(images[1:30], desc="Processing pages")):
        page_text = extract_text_from_image(img, prompt = prompt)
        page_text = page_text.replace(prompt,'')
        print(page_text)
        all_text += "\n" + page_text

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"Extraction complete. Text saved to {output_txt_path}")

if __name__ == "__main__":
    main()