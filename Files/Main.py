import numpy as np
from PIL import Image as im


def readBitmapFromFile(fileName):
    return np.asarray(im.open(fileName))


def writeBitmapToFile(bitmap, fileName):
    image = im.fromarray(np.uint8(bitmap))
    path = "Done/" + fileName
    image.save(path)


def calculateAveragesRGBColor(bitmap):
    rSum, gSum, bSum = 0, 0, 0
    for row in bitmap:
        for column in row:
            rSum += column[0]
            gSum += column[1]
            bSum += column[2]
    pixelsCount = len(bitmap) * len(bitmap[0])
    return [rSum / pixelsCount, gSum / pixelsCount, bSum / pixelsCount]


def main():
    fileName = "JGC0.jpg"
    print("Photos/" + fileName)
    bitmap = readBitmapFromFile("Photos/" + fileName)
    print(calculateAveragesRGBColor(bitmap))
    print(np.std(bitmap))
    writeBitmapToFile(bitmap, fileName)


if __name__ == '__main__':
    main()
