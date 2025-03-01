from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class image_parsing():
    def model_setup(self):
        model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2"
        ).to("cuda")

    def input_generation(self, image1):
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image1},
                    {"type": "text", "text": "Describe the information in the image in detail"}
                ]
            },
        ]

        # Prepare inputs
        inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device, dtype=torch.bfloat16)
        #inputs = self.processor(text=prompt, images=[image1], return_tensors="pt")
        #inputs = inputs.to(DEVICE)
        return inputs

