import torch
import base64
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load the model and it's processor
models = {
    "Qwen/Qwen2-VL-2B-Instruct-AWQ": Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto" # previously "auto" for torch_dtype
    ),
}

processors = {
    "Qwen/Qwen2-VL-2B-Instruct-AWQ": AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-AWQ"),
}



def image_to_base64(image):
    """
    Takes PIL Image and returns Base64-encoded string
    Args:
        image (PIL.Image.Image): The image to be converted to a Base64 string.

    Returns:
        str: The Base64-encoded string representation of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def run_example(image, text_input, system_prompt, model_id="Qwen/Qwen2-VL-2B-Instruct-AWQ"):
    """
    Generates a text response from a Base64-encoded image, text input, and system prompt using a specified vision-language model / multimodal LLM.
    Args:
        image (str): A Base64-encoded string of the image to be processed.
        text_input (str): The user-provided text input to be included in the prompt.
        system_prompt (str): A system prompt to guide the model's response.
        model_id (str, optional): The identifier of the model to be used. Defaults to 
                                  "Qwen/Qwen2-VL-2B-Instruct-AWQ".

    Returns:
        list of str: A list of generated text outputs from the model.
    """
    model = models[model_id].eval()
    processor = processors[model_id]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image}"},
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": text_input},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=512, 
        #do_sample=True,  # Enable sampling
        temperature=0.6,  # Adjust temperature for randomness
        #top_k=50,         # Use top-k sampling
        #top_p=0.95,        # Use top-p (nucleus) sampling
    )
    #Took: 4 min for 512, 7:15 min for 1024 tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print('OUTPUT_TEXT:', output_text)
    return output_text 
