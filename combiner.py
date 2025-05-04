import json
import os
import re

class PageDataCombiner:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.combined_data = {}

    def load_text_files(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith("_text.txt"):
                page_number = self._extract_page_number(filename)
                with open(os.path.join(self.folder_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    self._ensure_page_entry(page_number)
                    self.combined_data[page_number]['text'] = text

    def load_image_descriptions(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith("_image_descriptions.json"):
                with open(os.path.join(self.folder_path, filename), 'r', encoding='utf-8') as f:
                    image_data = json.load(f)
                    for img_filename, descriptions in image_data.items():
                        page_number = self._extract_page_number(img_filename)
                        self._ensure_page_entry(page_number)
                        self.combined_data[page_number].setdefault('images', {})[img_filename] = descriptions

    def _extract_page_number(self, filename):
        match = re.search(r'page_(\d+)', filename)
        return int(match.group(1)) if match else None

    def _ensure_page_entry(self, page_number):
        if page_number is not None and page_number not in self.combined_data:
            self.combined_data[page_number] = {}

    def combine(self):
        self.load_text_files()
        self.load_image_descriptions()
        return self.combined_data

    def save_combined_json(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.combined_data, f, indent=4, ensure_ascii=False)

# Example usage:
combiner = PageDataCombiner('./albania2')
combined = combiner.combine()
combiner.save_combined_json('./combined_output.json')