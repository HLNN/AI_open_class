import numpy
import cv2


class Inter:
    def __init__(self):
        self.factor = 2
        self.src = cv2.imread("Lena.jpg")
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

    def main(self):
        array = numpy.zeros((len(self.gray) // self.factor, len(self.gray[0]) // self.factor), dtype=numpy.uint8)
        for i in range(len(array)):
            for j in range(len(array[0])):
                array[i][j] = self.gray[int(i * self.factor)][int(j * self.factor)]
        dst = cv2.merge([array])
        cv2.imshow("src", self.gray)
        cv2.imshow("Inter", dst)
        cv2.waitKey(30000)


class Average:
    def __init__(self):
        self.factor = 2
        self.src = cv2.imread("Lena.jpg")
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

    def main(self):
        array = numpy.zeros((len(self.gray) // self.factor, len(self.gray[0]) // self.factor), dtype=numpy.uint8)
        for i in range(len(array)):
            for j in range(len(array[0])):
                if (i + 1) * self.factor < len(self.gray) and (j + 1) * self.factor < len(self.gray[0]):
                    a = self.gray[i * self.factor: (i + 1) * self.factor, j * self.factor: (j + 1) * self.factor]
                    array[i][j] = int(numpy.mean(a))
                else:
                    array[i][j] = self.gray[i * self.factor][j * self.factor]
        dst = cv2.merge([array])
        cv2.imshow("src", self.gray)
        cv2.imshow("Average", dst)
        cv2.waitKey(30000)


if __name__ == '__main__':
    inter = Inter()
    inter.main()
    average = Average()
    average.main()
