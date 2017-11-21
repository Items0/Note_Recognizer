import numpy as np
from PIL import Image as im


def readBitmapFromFile(fileName):
    return np.asarray(im.open(fileName))


def writeBitmapToFile(bitmap, fileName):
    image = im.fromarray(np.uint8(bitmap))
    path = "Done/" + fileName
    image.save(path)


def main():
    fileName = "JGC0.jpg"
    print("Photos/" + fileName)
    bitmap = readBitmapFromFile("Photos/" + fileName)
    writeBitmapToFile(bitmap, fileName)


if __name__ == '__main__':
    main()
