import fitz
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

def extract_from_pdf(pdf_path, output_folder, eval_page_nums=None):
    doc = fitz.open(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda" if torch.cuda.is_available() else "cpu")
    
    for page_num, page in enumerate(doc):
        if eval_page_nums and (page_num + 1) not in eval_page_nums:
            continue
        
        text_blocks = page.get_text("blocks")
        tables = page.find_tables()
        table_rects = [table.bbox for table in tables.tables] if tables and tables.tables else []
        
        filtered_text = "\n".join(block[4] for block in text_blocks if not any(fitz.Rect(block[:4]).intersects(rect) for rect in table_rects))
        with open(os.path.join(output_folder, f'page_{page_num+1}_text.txt'), 'w', encoding='utf-8') as f:
            f.write(filtered_text)
        
        if tables and tables.tables:
            table_data = [table.to_pandas().to_dict(orient="records") for table in tables.tables]
            with open(os.path.join(output_folder, f'page_{page_num+1}_tables.json'), 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=4)
        
        image_paths = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref, base_image = img[0], doc.extract_image(img[0])
            image_path = os.path.join(output_folder, f'page_{page_num+1}_img_{img_index+1}.{base_image["ext"]}')
            with open(image_path, "wb") as f:
                f.write(base_image["image"])
            image_paths.append(image_path)
        
        if image_paths:
            extract_image_descriptions(image_paths, output_folder, page_num+1, processor, model)
    
    print(f"Extraction completed. Check '{output_folder}' for output files.")

def extract_image_descriptions(image_paths, output_folder, page_num, processor, model):
    descriptions = {}
    prompt = "Describe the image content in detail."
    
    for image_path in tqdm(image_paths, desc=f"Processing images for page {page_num}"):
        image = Image.open(image_path).convert("RGB")
        descriptions[os.path.basename(image_path)] = process_image(image, prompt, processor, model)
    
    with open(os.path.join(output_folder, f'page_{page_num}_image_descriptions.json'), 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=4)

def process_image(image, prompt, processor, model):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.float16)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Example usage
extract_from_pdf("C:/Users/Rudy/Desktop/UN-run/national_drr_strategy_albania.pdf", "C:/Users/Rudy/Desktop/UN-run/albania/", eval_page_nums = [11, 28, 41])
