from pdf2image import convert_from_path
from transformers.image_utils import load_image
from image_parsing import image_parsing

pages = convert_from_path('./Albania/National_Adaptation_Plan_Albania.pdf', 500)

for count, page in enumerate(pages):
    page.save(f'./Albania/out_{count}.jpg', 'JPEG')

for i in range(12,30):
    # Load images
    image1 = "./Albania/out_"+str(i)+".jpg"

    image_parsing_obj = image_parsing()
    image_parsing_obj.model_setup()
    inputs = image_parsing_obj.input_generation(image1)

    # Generate outputs
    generated_ids = image_parsing_obj.model.generate(**inputs, do_sample=False, max_new_tokens=500)
    generated_texts = image_parsing_obj.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])
