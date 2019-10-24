import os, argparse
import skimage
from skimage import io, transform, util
import PIL
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('f', metavar='foldername', help='takes the name of the root folder to be opened')
args = parser.parse_args()

for files in os.listdir(args.f):
    if files != ".DS_Store" :
        img = Image.open(args.f + files)
        img = img.resize((700, 600))
        img = img.rotate(12)
        img = img.crop((30, 30, 650, 570))
        img.save(files)
