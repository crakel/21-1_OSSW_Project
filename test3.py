from google.cloud import vision
from google.cloud.vision_v1 import types

import io
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:/Users/ALLxNEW/Desktop/key.json'

client = vision.ImageAnnotatorClient()
path ='C:/Users/ALLxNEW/Desktop/test4.png'

file_name=os.path.join(os.path.dirname(__file__), path)
with io.open(file_name,'rb') as image_file:
    content=image_file.read()

image=types.Image(content=content)
# with io.open(path, 'rb') as image_file:
#     content = image_file.read()

price_candidate = []
card_number_candidate = []
date_candidate = []

response = client.text_detection(image=image)
texts = response.text_annotations
print('Texts:')

for text in texts:
    content = text.description
    content = content.replace(',' , '')
    print('\n"{}"'.format(content))





if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))
