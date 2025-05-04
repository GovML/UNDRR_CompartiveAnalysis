# run after running pdf_parse_eval.py and combiner.py

import json
from qa.deepseek_client import DeepSeekClient  # Adjust if needed for your environment

class PageQA:
    def __init__(self, json_path, model_name="deepseek-r1:7b", temperature=0.6):
        self.data = self._load_json(json_path)
        self.client = DeepSeekClient(model_name=model_name, temperature=temperature)

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_context(self):
        context_parts = []
        for page, content in sorted(self.data.items(), key=lambda x: int(x[0])):
            page_header = f"\n--- Page {page} ---\n"
            text = content.get("text", "")
            images = content.get("images", {})
            img_captions = "\n".join(
                [f"{img}: {desc.get('<MORE_DETAILED_CAPTION>', '')}" for img, desc in images.items()]
            )
            context_parts.append(f"{page_header}\n{text}\n{img_captions}")
        return "\n".join(context_parts)

    def ask(self, question: str) -> str:
        context = self.get_context()
        prompt = f"""You are an assistant answering questions about a document. You must cite your source from within the content of the document.
Here is the content of the document:

{context}

Question: {question}
Answer:"""
        response = self.client.generate(prompt)
        return response['text'] if isinstance(response, dict) and 'text' in response else str(response)

# Example usage (remove or comment this out when importing elsewhere):
if __name__ == "__main__":
    qa = PageQA("combined_output.json")
    question = "What sources of funding does this document outline?"
    print("Q:", question)
    print("A:", qa.ask(question))