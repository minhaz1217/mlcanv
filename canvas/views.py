from django.shortcuts import render
from django.http import HttpResponse

from . import predict
import re
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
def myTest(request):
    return render(request, "main.html")

def saveImage(str):
    '''
    imgstr = re.search(r'base64,(.*)', str).group(1)
    output = open('output.png', 'wb')
    img = base64.b64decode(imgstr)
    output.write(img)
    output.close()
    '''
    '''
    imgstr = re.search(r'base64,(.*)', str).group(1)
    im = Image.open(BytesIO(base64.b64decode(imgstr)))
    im.save("mine.png", "PNG")
    '''
    image_data = re.sub('^data:image/.+;base64,', '', str)
    im = Image.open(BytesIO(base64.b64decode(image_data)))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(im)
    #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('mine.png')
    plt.close()
    return 1


def showCanvas(request):
    if(request.method == "POST" ):
        mine = request.POST['myHidden']
        test2 = saveImage(mine)
        test3 = -1
        test3 = predict.predictThis()
        context = {
            'test' : mine,
            'test2' : test2,
            'test3' : test3
        }
        return render(request, "canvas.html", context)
    else:
        return render(request, "canvas.html")