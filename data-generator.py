import os
from os import listdir
from os.path import isfile, join
import pathlib
import string
from PIL import Image, ImageDraw, ImageFont

data_dir = 'data'

for l in string.ascii_uppercase:
    letter_dir = os.path.join(data_dir, l)
    pathlib.Path(letter_dir).mkdir(parents=True, exist_ok=True)


def generateLetterImage(letter, font, path):
    img_width, img_height = (256, 256)
    img = Image.new('RGB', (img_width, img_height))
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(font, 200)
    text_width, text_height = d.textsize(letter, fnt)
    d.text(((img_width - text_width)/2, (img_height - text_height)/2),
           letter, (255, 255, 255), fnt)
    img.thumbnail([28, 28], Image.ANTIALIAS)
    img.save(path)


fonts_dir = ['fonts']
fonts = [os.path.join(dir, font) for dir in fonts_dir for font in os.listdir(
    dir) if font.endswith(('ttf', 'otf'))]


onlyfiles = [os.path.join('fonts', f) for f in listdir('fonts') if isfile(join('fonts', f))]


# print(onlyfiles)

for letter in string.ascii_uppercase:
    for i, font in enumerate(onlyfiles):
        generateLetterImage(letter, font, os.path.join(
            data_dir, letter, f'{letter}.{i}.png'))

