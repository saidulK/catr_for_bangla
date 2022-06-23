import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import random
import os
import json

with open('captions.json','r',encoding="utf8") as file:
    captions = json.load(file)

location = 'G:\\Pioneer Alpha\\Task 3'
image_names = sorted([int(name.split('.')[0]) for name in os.listdir(os.path.join(location,'images','images'))])
image_names = [str(name)+'.png' for name in image_names]

for i in range(1,5):
    
    print(img_name)
    print('\n',1,"       :",end='')
    for sentence in captions[i-1]['caption']:
            print(sentence)
    print('\n',2,"       :",end='')
    for sentence in captions[i]['caption']:
            print(sentence)
    print('\n',3,"       :",end='')
    for sentence in captions[i+1]['caption']:
            print(sentence)
    img_name = image_names[i]
    img = mpimg.imread(os.path.join('G:\\Pioneer Alpha\\Task 3\\images\\images',img_name))
    imgplot = plt.imshow(img)
    plt.show()

    a = input()
