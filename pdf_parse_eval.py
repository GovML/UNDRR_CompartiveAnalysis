import fitz
import os
import math
import json
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class PDFExtractor:
    def __init__(self, device=None):
        #self.device = device or ("mps" if torch.cuda.is_available() else "cpu")
        self.device = "mps"
        #self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.torch_dtype = torch.float16
        print(f'Loading onto {self.device}')
        print("Loading Florence-2-large model and processor...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            revision="main"
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            revision="main"
        )
        print("Model and processor loaded successfully.")

    def process_pdfs(self, pdf_list):
        """
        Processes multiple PDFs, extracting text, tables, and images.
        :param pdf_list: List of tuples (pdf_path, output_folder, optional list of page numbers)
        """
        for pdf_path, pdf_output_folder, eval_pages in pdf_list:
            print(f"\nProcessing PDF: {pdf_path}")

            # Ensure unique output directory per PDF
            os.makedirs(pdf_output_folder, exist_ok=True)
            self._extract_from_pdf(pdf_path, pdf_output_folder, eval_pages)

    def _extract_from_pdf(self, pdf_path, output_folder, eval_page_nums=None):
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            if eval_page_nums and (page_num + 1) not in eval_page_nums:
                continue

            print(f"\nProcessing page {page_num + 1} of {os.path.basename(pdf_path)}...")

            self._extract_text(page, page_num, output_folder)
            self._extract_tables(page, page_num, output_folder)
            image_paths = self._extract_images(page, page_num, output_folder)

            if image_paths:
                self._extract_image_descriptions(image_paths, page_num, output_folder)

        print(f"Completed processing {os.path.basename(pdf_path)}. Check '{output_folder}' for results.")

    def _extract_text(self, page, page_num, output_folder):
        print("Extracting text...")
        text_blocks = page.get_text("blocks")
        text_content = "\n".join(block[4] for block in text_blocks)

        output_path = os.path.join(output_folder, f'page_{page_num+1}_text.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"Text saved to {output_path}")

    def _extract_tables(self, page, page_num, output_folder):
        print("Extracting tables...")
        tables = page.find_tables()

        if tables and tables.tables:
            table_data = [table.to_pandas().to_dict(orient="records") for table in tables.tables]
            output_path = os.path.join(output_folder, f'page_{page_num+1}_tables.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=4)
            print(f"Tables saved to {output_path}")
        else:
            print("No tables found.")

    def _extract_images(self, page, page_num, output_folder):
        print("Extracting images...")
        image_paths = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref, base_image = img[0], page.parent.extract_image(img[0])
            image_path = os.path.join(output_folder, f'page_{page_num+1}_img_{img_index+1}.{base_image["ext"]}')
            with open(image_path, "wb") as f:
                f.write(base_image["image"])
            image_paths.append(image_path)

        print(f"Extracted {len(image_paths)} images.")
        return image_paths

    def _extract_image_descriptions(self, image_paths, page_num, output_folder):
        print(f"\nGenerating descriptions for {len(image_paths)} images on page {page_num + 1}...")
        descriptions = {}
        prompt = "<MORE_DETAILED_CAPTION>"

        for image_path in tqdm(image_paths, desc=f"Processing images for page {page_num+1}"):
            print(f"Processing {image_path}...")

            image = Image.open(image_path)
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

            print("Generating description...")
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False
            )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
            descriptions[os.path.basename(image_path)] = parsed_answer

            print(f"Generated description: {parsed_answer}")

        output_path = os.path.join(output_folder, f'page_{page_num+1}_image_descriptions.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, indent=4)

        print(f"Image descriptions saved to {output_path}")


class JsonCombiner:
    def __init__(self, device=None):
        #self.device = device or ("mps" if torch.cuda.is_available() else "cpu")
        self.device = "mps"
        #self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.torch_dtype = torch.float16

    def jsonCombineCreateDocList(self, input_folder):
        self.list_docs = []
        for file in os.listdir(input_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") or filename.endswith(".json"):
                self.list_docs.append(file)
        self.list_docs.sort()

    def createJsonFromListDocs(self, input_folder, output_folder=None):
        data = []
        for i in range(len(self.list_docs)):
            if ".txt" in self.list_docs[i]:
                with open(input_folder + '/' + self.list_docs[i]) as inp:
                    for line in inp:
                        data.append(line)
            if ".json" in self.list_docs[i]:
                if "image" in self.list_docs[i]:
                    with open(input_folder + '/' + self.list_docs[i]) as json_data:
                        d = json.load(json_data)
                        for key in d.keys():
                            data.append(d[key]["<MORE_DETAILED_CAPTION>"])
                else:
                    with open(input_folder + '/' + self.list_docs[i]) as json_data:
                        d = json.load(json_data)
                        for j in range(len(d[0])):
                            for key in d[0][j].keys():
                                data.append(str(key) + "-" + str(d[0][j][key]))
            #print(i)
            if (i % 10 == 0 and i != 0):
                combined_text = "".join(data)
                output_path = os.path.join(output_folder, f'page_combined_txt_{(i/10)+1}.txt')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                #print(f"Text saved to {output_path}")
                data = []
        combined_text = "".join(data)
        output_path = os.path.join(output_folder, f'page_combined_txt_{math.floor(i/10) + 1}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)





# Example Usage
pdfs_to_process = [
    ("./albania/national_drr_strategy.pdf", 
     "./albania/", [11, 28, 41]),
     
    # ("C:/Users/Rudy/Desktop/UN-run/another_document.pdf", 
    #  "C:/Users/Rudy/Desktop/UN-run/another_output/", None)  # Process all pages
]

#extractor = PDFExtractor()
#extractor.process_pdfs(pdfs_to_process)


combiner = JsonCombiner()
combiner.jsonCombineCreateDocList("./albania/")
combiner.createJsonFromListDocs("./albania", "./albania_outputs/")
