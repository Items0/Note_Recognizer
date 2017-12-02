import numpy as np
from PIL import Image as im
import PIL.ImageStat as imageStat  # fajna clasa analizująca  imageStat.Stat(Image)._get[co chcę (mean, stddev ...)]
import matplotlib.pyplot as plt
import skimage
from skimage.filters.edges import convolve
import skimage.morphology as morph
from skimage.morphology import square
from skimage.feature import match_template, peak_local_max
from skimage import io, data, draw, measure
from skimage.draw import circle_perimeter, set_color
import math

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

IDENT_PARAM = 0.52  # <-1,1>


def readBitmapFromFile(fileName):
    path = "Photos/" + fileName + ".jpg"
    image = im.open(path)
    image = image.resize((int(image.size[0] / np.sqrt(6)), int(image.size[1] / np.sqrt(6))))
    return np.asarray(image)


def writeBitmapToFile(bitmap, fileName):
    image = im.fromarray(np.uint8(bitmap))
    path = "Done/" + fileName + FILE_SUFIX + ".jpg"
    image.save(path)


def filterImage(image):
    print("Obrabiam obrazek")
    image = np.abs(convolve(image, MASK_MEAN))
    image = skimage.filters.sobel(image)
    image = image > skimage.filters.threshold_li(image)
    image = morph.erosion(image)
    image = morph.dilation(image)
    image, blob = toHorizontalLevel(image)
    im.fromarray(np.uint8(image * 255)).show()
    im.fromarray(np.uint8(blob * 255)).show()

    starts, stops = detectStartsAndEndsBlobs(blob)
    imageParts = divideImageOnParts(image, starts, stops)
    for i in range(len(imageParts)):
        img = im.fromarray(np.uint8(imageParts[i]) * 255)
        img.save(str(i) + ".jpg")
    return image


def toHorizontalLevel(image):
    print("Poziomuję obrazek")
    while True:
        blob = makeBlobs(image)

        image2 = np.asarray(im.fromarray(np.uint8(blob)).resize((int(len(blob) / 20), (int(len(blob[0]) / 20)))))
        print((len(image2) * len(image2[0])), " ", (len(image2) * len(image2[0]) / 100))
        zerosMatrix = np.zeros((len(image2), len(image2[0])))
        limit = len(image2) * len(image2[0]) / 100
        detectOneStaff(image2, zerosMatrix, limit)

        xPositions, yPositions = prepareDataToRegression(zerosMatrix)
        aParameter, bParameter = np.polyfit(xPositions, yPositions, 1)
        print(aParameter)
        print(bParameter)
        angle = np.arctan(aParameter) * 188 / np.pi
        print("angle ", angle)

        image = np.asarray(im.fromarray(np.uint8(image)).rotate(angle))
        blob = np.asarray(im.fromarray(np.uint8(blob)).rotate(angle))
        if np.abs(angle) < 10:
            break
    return image, blob

def makeBlobs(image):
    print("Robię blob'y")
    blob = morph.dilation(image, square(30))
    return blob


def rowContainWhite(row):
    for i in range(len(row)):
        if row[i] == 1:
            return True
    return False


def detectStartsAndEndsBlobs(image):
    print("Szukam początków i końców blob'ów")
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
    print("Dzielę obrazek na części")
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


def detectOneStaff(blob, zerosMatrix, limit):
    print("Szukam bloba z pięciolinią")
    for y in range(len(blob)):
        for x in range(len(blob[0])):
            if (blob[y, x] == 1 and zerosMatrix[y, x] == 0):
                counter = markOneShape(blob,zerosMatrix, x, y, 0)
                if (counter > limit):
                    return



def markOneShape(blob, zerosMatrix, x, y, counter):
    zerosMatrix[y, x] = 1
    if y + 1 < len(blob) and blob[y + 1, x] == 1 and zerosMatrix[y + 1, x] == 0:
        counter = markOneShape(blob, zerosMatrix, x, y + 1, counter+1)
    if y - 1 >= 0 and blob[y - 1, x] == 1 and zerosMatrix[y - 1, x] == 0:
        counter = markOneShape(blob, zerosMatrix, x, y - 1, counter+1)
    if x + 1 < len(blob[0]) and blob[y, x + 1] == 1 and zerosMatrix[y, x + 1] == 0:
        counter = markOneShape(blob, zerosMatrix, x + 1, y, counter+1)
    if x - 1 >= 0 and blob[y, x - 1] == 1 and zerosMatrix[y, x - 1] == 0:
        counter = markOneShape(blob, zerosMatrix, x - 1, y, counter+1)
    return counter


def prepareDataToRegression(image):
    print("Przygotowuję dane do regresji")
    xPositions = []
    yPositions = []
    for y in range(len(image)):
        for x in range(len(image[0])):
            if image[y, x] == 1:
                xPositions.append(x)
                yPositions.append(y)
    return xPositions, yPositions


def calculateRegressionFromMachineLearning(xPositions, yPositions):
    print("Liczę regresję z machine learning")
    # todo mocno zmniejszyć dane wejściowe
    aParameter = 0.0  # inicjalizuj wagi
    bParameter = 0.0  #
    learning_rate = 0.0001  # stala uczenia
    maxIteration = 10000  # liczba iteracji

    dividor = 1000

    for i in range(maxIteration):
        if i % dividor == 0:
            print("Skończyłem ", str(i / dividor), "%")
        toB = 0.0
        toA = 0.0
        for j in range(len(xPositions)):
            toB += aParameter * xPositions[j] + bParameter - yPositions[j]
            toA += (aParameter * xPositions[j] + bParameter - yPositions[j]) * xPositions[j]
        aParameter = aParameter - learning_rate * (1 / len(xPositions)) * toA
        bParameter = bParameter - learning_rate * (1 / len(xPositions)) * toB
    return aParameter, bParameter


def findElement(myImage, myElement, myCopy, ax, myColor):
    result = match_template(myImage, myElement)
    tab = peak_local_max(result, 20, threshold_rel=IDENT_PARAM)
    for el in tab:
        y = el[0]
        x = el[1]
        height, width = myElement.shape
        myImage[y:y + height, x:x + width] = 0
        # print (y)
        # print (x)
        rect = plt.Rectangle((x, y), width, height, edgecolor=myColor, fill=False)
        ax.add_patch(rect)
    return myImage, myCopy

    # io.imshow(myImage)
    # plt.show()

    # myElement = io.imread("Patterns/full_note.jpg", as_grey=True)
    # for i in range(0, 4):
    #     myImage, myCopy = findElement(myImage, myElement, myCopy)
    # io.imshow(myCopy)
    # plt.show()


def findSth(elements):
    for i in elements:
        mean = round(np.mean(i), 3)
        std = round(np.std(i), 3)
        var = round(np.var(i), 3)
        print(mean, std, var)


def loadElements(myNames):
    elements = []
    for i in myNames:
        elements.append(io.imread("Patterns/" + i + ".jpg", as_grey=True))
    return elements


def main():
    myNames = ['chord3', 'chord2', 'trebleClef', 'bassClef', 'eighthNote', 'quarterNote', 'wholeNote']
    frameColor = ['yellow', 'coral', 'b', 'r', 'm', 'c', 'g']

    fileName = "GGC0"
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    elements = loadElements(myNames)
    # line = io.imread("Patterns/line.jpg", as_grey=True)
    # findSth(elements)
    myImage = io.imread("Photos/JGC3.jpg", as_grey=True)
    myCopy = io.imread("Photos/JGC0.jpg", as_grey=True)
    myImage = filterImage(myImage)

    for i in range(len(elements)):
        myImage, myCopy = findElement(myImage, elements[i], myCopy, ax, frameColor[i])
    io.imshow(myCopy)
    plt.show()


if __name__ == '__main__':
    main()
