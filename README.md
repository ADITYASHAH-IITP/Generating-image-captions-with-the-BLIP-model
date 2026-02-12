# Generating-image-captions-with-the-BLIP-model
Implement an image captioning tool using the BLIP model from Hugging Face's Transformers. Use Gradio to provide a user-friendly interface for your image captioning application. Adapt the tool for real-world business scenarios, demonstrating its practical applications


#set up the environment and dependencies for this project. 
1. Open up a new terminal
2. Create a Python virtual environment and install Gradio using the following commands in the terminal:
  1. pip3 install virtualenv 
  2. virtualenv my_env # create a virtual environment my_env 
  3. source my_env/bin/activate # activate my_env


Install the required libraries in the environment:

## Installing required libraries in my_env
pip install langchain==0.1.11 gradio==5.23.2 transformers==4.38.2 bs4==0.0.2 requests==2.31.0 torch==2.2.1

"AutoProcessor" and "BlipForConditionalGeneration" are components of the BLIP model, which is a vision-language model available in the Hugging Face Transformers library.

AutoProcessor : This is a processor class that is used for preprocessing data for the BLIP model. It wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor. This means it can handle both image and text data, preparing it for input into the BLIP model.

Note: A tokenizer is a tool in natural language processing that breaks down text into smaller, manageable units (tokens), such as words or phrases, enabling models to analyze and understand the text.

BlipForConditionalGeneration : This is a model class that is used for conditional text generation given an image and an optional text prompt. In other words, it can generate text based on an input image and an optional piece of text. This makes it useful for tasks like image captioning or visual question answering, where the model needs to generate text that describes an image or answer a question about an image.

Step 1: Import your required tools from the transformers library
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

## Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

Step 2: Load and Preprocess an Image
The Python Imaging Library, PIL, is used to open the image file and convert it into an RGB format which is suitable for the model.
## Load your image, DON'T FORGET TO WRITE YOUR IMAGE NAME
img_path = "YOUR IMAGE NAME.jpeg"
## Convert it into an RGB format 
Next, the pre-processed image is passed through the processor to generate inputs in the required format. The return_tensors argument is set to "pt" to return PyTorch tensors.
image = Image.open(img_path).convert('RGB')

text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

## Generate a caption for the image
The argument max_length=50 specifies that the model should generate a caption of up to 50 tokens in length.
outputs = model.generate(**inputs, max_length=50)

## Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
## Print the caption
print(caption)

python3 image_cap.py


# Implement image captioning app with Gradio
Step 1: Set up the environment
Make sure you have the necessary libraries installed. Run pip install gradio transformers Pillow to install Gradio, Transformers, and Pillow.
Create a new Python file and call it image_captioning_app.py


Step 2: Load the pretrained model
Load the pretrained processor and model:
processor = # write your code here
model = # write your code here

Step 3: Define the image captioning function
Define the caption_image function that takes an input image and returns a caption.

Step 4: Create the Gradio interface
Use the gr.Interface class to create the web app interface:

Step 5: Launch the Web App
Start the web app by calling the launch() method.

Step 6: Run the application

Save the complete code to a Python file, for example, image_captioning_app.py.
Open a terminal or command prompt, navigate to the directory where the file is located, and run the command
python3 image_captioning_app.py


Demo:
<img width="1069" height="606" alt="image" src="https://github.com/user-attachments/assets/3295a181-4fc5-4b98-b11c-e15b557f179f" />


## How image captioning helps a business 
See automate_url_captioner.py

## Image captioning with BLIP2
img_captioner_blip2.py


# Deploy your app with Code Engine
Container images and containers
Code Engine lets you run your apps in containers on IBM Cloud. A container is an isolated environment or place where an application can run independently. Containers can run anywhere, such as on operating systems, virtual machines, developer's machines, physical servers, and so on. This allows the containerized application to run anywhere as well and the isolation mechanism ensures that the running application will not interfere with the rest of the system.

Containers are created from container images. A container image is basically a snapshot or a blueprint that indicates what will be in a container when it runs. Therefore, to deploy a containerized app, you first need to create the app's container image.

## Creating the container image
The files required to deploy your app in a container are as follows:

You can use the Gradio framework for generating the user interface of the app,for example, the Python script that contains the code to create and launch the gradio.Interface can be named as demo.py.
The source code of the app has its dependencies, such as libraries that the code uses. Hence, you need a requirements.txt file that specifies all the libraries the source code depends on.
You need a file that shows the container runtime the steps for assembling the container image, named as Dockerfile.

Open a terminal and make a new directory myapp for storing the files and go into the directory with the following command and create the files:
 mkdir myapp
 cd myapp
 touch demo.py Dockerfile requirements.txt

 Step 1: Creating requirements.txt
  We can install all of the dependencies into your environment at once with the command pip3 install -r requirements.txt.

 Step 2: Creating demo.py
  Creating a simple Gradio web application - Check demo.py
  
 Step 3: Creating Dockerfile
  The Dockerfile is the blueprint for assembling a container image - Check Dockerfile 
  What does the Dockerfile do?
  FROM python:3.10
    Docker images can be inherited from other images. Therefore, instead of creating your own base image, you will use the official Python image python:3.10 that already has all the tools and packages that you need to run a Python application.
  
  WORKDIR /app
    To facilitate the running of your commands, let's create a working directory /app. This instructs Docker to use this path as the default location for all subsequent commands. By creating the directory, you do not have to type out full file paths but can use    relative paths based on the working directory.
  
  COPY requirements.txt requirements.txt
    Before you run pip3 install, you need to get your requirements.txt file into your image. You can use the COPYcommand to transfer the contents. The COPYcommand takes two parameters. The first parameter indicates to the Docker what file(s) you would like to copy into the image. The second parameter indicates to the Docker the location where the file(s) need to be copied. You can move the requirements.txt file into your working directory /app.
  
  RUN pip3 install –no-cache-dir -r requirements.txt
    Once you have your requirements.txt file inside the image, you can use the RUN command to execute the command pip3 install --no-cache-dir -r requirements.txt. This works exactly the same as if you were running the command locally on your machine, but this time the modules are installed into the image.
  
  COPY
    At this point, you have an image that is based on Python version 3.10 and you have installed your dependencies. The next step is to add your source code to the image. You will use the COPY command just like you did with your requirements.txt file above to copy everything in your current working directory to the file system in the container image.
  
  CMD ["python", "demo.py"]
    Now, you have to indicate to the Docker what command you want to run when your image is executed inside a container. You use the CMD command. Docker will run the python demo.py command to launch your app inside the container.

