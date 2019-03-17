from PIL import Image
import scipy.misc
from matplotlib.pyplot import imshow

def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)

file_in = '/Users/enjlife/deeplearning-practice/style-transfer/content_images/content4.jpeg'
file_out = '/Users/enjlife/deeplearning-practice/style-transfer/content_images/content5.jpg'

produceImage(file_in, 400, 300, file_out)
# style_image = scipy.misc.imread("/Users/enjlife/deeplearning-practice/style-transfer/content_images/content3.jpg")
# imshow(style_image)