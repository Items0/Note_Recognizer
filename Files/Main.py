import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
from PIL import Image as im
from skimage.feature import match_template
from skimage import io, data, draw
from skimage.draw import circle_perimeter
import PIL.ImageStat as imageStat  # fajna clasa analizująca  imageStat.Stat(Image)._get[co chcę (mean, stddev ...)]
import matplotlib.pyplot as plt
from skimage.filters.edges import convolve
import skimage.morphology as morph
from skimage.morphology import square

MULTIPLE_STD_PARAM = 2.0
FILE_SUFIX = ""
MASK_EDGE_HORIZONTAL = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])
MASK_EDGE_VERTICAL = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]]) / 8
MASK_EDGE_LAPLACE = np.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
MASK_MEAN = np.array([[1, 1, 1],
                      [1, 2, 1],
                      [1, 1, 1]]) / 20
MASK_DILATATION = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])


def readBitmapFromFile(fileName):
    path = "Photos/" + fileName + ".jpg"
    image = im.open(path)
    image = image.resize((int(image.size[0] / np.sqrt(6)), int(image.size[1] / np.sqrt(6))))
    return np.asarray(image)


def writeBitmapToFile(bitmap, fileName):
    image = im.fromarray(np.uint8(bitmap))
    path = "Done/" + fileName + FILE_SUFIX + ".jpg"
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


def thresholding(bitmap, avegaresRGB, std=0):
    threshBitmap = []
    std *= MULTIPLE_STD_PARAM
    rThresh = avegaresRGB[0] - std if avegaresRGB[0] - std > 0 else 0
    gThresh = avegaresRGB[1] - std if avegaresRGB[1] - std > 0 else 0
    bThresh = avegaresRGB[2] - std if avegaresRGB[2] - std > 0 else 0
    print(avegaresRGB)
    print(rThresh, " ", gThresh, " ", bThresh)
    print(std)
    for row in bitmap:
        threshRow = []
        for cell in row:
            r = 255 if rThresh < cell[0] else 0
            g = 255 if gThresh < cell[1] else 0
            b = 255 if bThresh < cell[2] else 0
            threshRow.append([r, g, b])
        threshBitmap.append(threshRow)
    return threshBitmap


def findElement(myImage, myElement, myCopy):
    result = match_template(myImage, myElement)
    y, x = np.unravel_index(np.argmax(result), result.shape)
    height, width = myElement.shape
    myImage[y:y + height, x:x + width] = 0
    print(y)
    print(x)
    rr, cc = circle_perimeter(math.ceil(y + height / 2), math.ceil(x + width / 2), min(height, width))
    myCopy[rr, cc] = 1
    return myImage, myCopy


def createBitmapWithMask(bitmap, mask):
    return np.abs(convolve(bitmap, mask[:, :, None]))


def edgeDetect(bitmap):
    bitmap = createBitmapWithMask(bitmap, MASK_MEAN)
    bitmap = createBitmapWithMask(bitmap, MASK_EDGE_LAPLACE)
    return bitmap


def createBitmapWithMask2D(bitmap, mask):
    return np.abs(convolve(bitmap, mask[:, :]))


def edgeDetect2D(bitmap):
    bitmap = createBitmapWithMask2D(bitmap, MASK_MEAN)
    bitmap = createBitmapWithMask2D(bitmap, MASK_EDGE_LAPLACE)
    return bitmap


def detectVerticalEdge(bitmap):
    return createBitmapWithMask(bitmap, MASK_EDGE_VERTICAL)


def detectLineVertical(verticals, bitmap):
    detectedBitmap = []
    print(len(verticals))
    print(len(verticals[0]))
    for y in range(len(verticals)):
        detectedRow = []
        for x in range(len(verticals[0])):
            avgColor = (bitmap[y, x, 0] + bitmap[y, x, 1] + bitmap[y, x, 2]) / 3
            if avgColor > 100:
                detectedRow.append([255, 255, 255])
            else:
                detectedRow.append([0, 0, 0])
        detectedBitmap.append(detectedRow)
    writeBitmapToFile(detectedBitmap, "test")


def filterImage(image):
    image = np.abs(convolve(image, MASK_MEAN))
    image = skimage.filters.sobel(image)
    image = image > skimage.filters.threshold_li(image)
    image = morph.erosion(image)
    image = morph.dilation(image)
    blob = makeBlobs(image)
    starts, stops = detectStartsAndEndsBlobs(blob)
    imageParts = divideImageOnParts(image, starts, stops)
    for part in imageParts:
        im.fromarray(np.uint8(part) * 255).show()
    return image


def makeBlobs(image):
    blob = morph.dilation(image, square(30))
    return blob


def rowContainWhite(row):
    for i in range(len(row)):
        if row[i] == 1:
            return True
    return False


def detectStartsAndEndsBlobs(image):
    starts = []
    ends = []
    isBlob = False
    counter = 0
    for row in image:
        whiteInRow = rowContainWhite(row)
        if not isBlob and whiteInRow:
            starts.append(counter)
            isBlob = True
        if isBlob and not whiteInRow:
            ends.append(counter)
            isBlob = False
        counter += 1
    return starts, ends


def divideImageOnParts(image, starts, stops):
    parts = []
    part = []
    rewrite = False
    for i in range(len(image)):
        if not rewrite and i in starts:
            rewrite = True
            part = []
        if rewrite and i in stops:
            rewrite = False
            parts.append(part)
        if rewrite:
            part.append(image[i] * 1)
    return parts


def main():
    '''fileName = ["GGC0", "GGC3", "GGO3", "GPN3", "GPN6", "GPO0", "JBO0", "JBO3", "JGC0", "JGC3", "JPC6",
    "JPN3", "JPO3", "NBO0", "NBO6", "NGC3", "NPN0", "PBO0", "PGC0", "PGO3", "PPC3", "PPO3"]

    for i in fileName:
        makeImage(i)'''

    # fileName = "JGC0"
    # makeImage(fileName)
    fig = plt.figure(figsize=(15, 10))
    myImage = io.imread("Photos/JGC0.jpg", as_grey=True)
    myCopy = io.imread("Photos/JGC0.jpg", as_grey=True)

    myImage = filterImage(myImage)

    # myImage = skimage.filters.median(myImage)
    # myImage = skimage.filters.sobel(myImage)


    # myCopy = skimage.filters.laplace(myCopy)


    # io.imshow(myImage)
    # plt.show()

    # myElement = io.imread("Patterns/full_note.jpg", as_grey=True)
    # for i in range(0, 4):
    #     myImage, myCopy = findElement(myImage, myElement, myCopy)
    # io.imshow(myCopy)
    # plt.show()


if __name__ == '__main__':
    main()
