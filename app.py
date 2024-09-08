import base64
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template 
from functions import image_to_base64, run_example


app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    
    image = data.get('image')
    text_input = data.get('text_input')

    #     # Check file size
    #     file_size = os.fstat(file.fileno()).st_size 
    #     max_size = 200 * 1024 * 1024  # 200MB in bytes
    #     if file_size > max_size:
    #         responses.append({'status': False, 'message': f'File {file.filename} exceeds maximum allowed size (200MB)'})
    #         continue
    #     file_size_kb = file_size / 1024

    # decoding the base64 image
    image_data = base64.b64decode(image)

    # converting the base64 image to PIL Image
    image = Image.open(BytesIO(image_data))

    # converting the PIL Image back to base64 to pass to the model
    image_data_base64 = image_to_base64(image)

    system_prompt = """
    Generate atleast 5 detailed test cases with respect to the given input image and text content.
    Output should describe a detailed, step-by-step guide on how to test each identified feature.
    
    Each test case should include:
    1) Test Case ID: A unique ID for each test case 
    2) Description: A few lines describing what the test case is about.
    3) Pre-conditions: What needs to be set up or ensured before testing.
    4) Testing Steps: Clear, step-by-step instructions on how to perform the test.
    5) Expected Result: What should happen if the feature works correctly.
    """

    model_id = data.get('model_id', 'Qwen/Qwen2-VL-2B-Instruct-AWQ', )

    # Initial prompt to the model
    initial_output = run_example(image_data_base64, text_input, system_prompt, model_id)
    
    refined_output = initial_output
    for _ in range(1): 
        refinement_prompt = f"""
        Refine the following test cases instructions.
          Ensure they are specific, comprehensive, and cover all aspects of the features shown in the image and described in the text input.
          Mention the Feature name with the Test Case ID.
          
        
        Test cases instructions:
        {refined_output}
        """
        #Do not remove anything. Only add and keep the previous content.
        #Below are the test cases instructions provided, please refine them further.
        #Add to the existing test case instructions only, especially the Description.
        refined_output = run_example(image_data_base64, text_input, refinement_prompt, model_id)
    

    return jsonify({
        "output_text": refined_output
    })



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9000, debug=True)
