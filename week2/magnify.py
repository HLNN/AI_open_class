import cv2
import numpy


class Nearest:
    def __init__(self):
        self.factor = 3
        self.src = cv2.imread("Lena.jpg")
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

    def main(self):
        array = numpy.zeros((len(self.gray) * self.factor, len(self.gray[0]) * self.factor), dtype=numpy.uint8)
        for i in range(len(array)):
            for j in range(len(array[0])):
                array[i][j] = self.gray[int(i / self.factor)][int(j / self.factor)]
        dst = cv2.merge([array])
        cv2.imshow("src", self.gray)
        cv2.imshow("Nearest", dst)
        cv2.waitKey(30000)


class Linear:
    def __init__(self):
        self.factor = 3
        self.src = cv2.imread("Lena.jpg")
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

    def inter(self, i, j, axis):
        if i // self.factor + 1 < len(self.gray) and j // self.factor + 1 < len(self.gray[0]):
            if axis % 2 == 1:
                if j % self.factor == 0:
                    return self.gray[i // self.factor][j // self.factor]
                a = (j % self.factor)
            else:
                if i % self.factor == 0:
                    return self.gray[i // self.factor][j // self.factor]
                a = (i % self.factor)
            p1 = self.gray[i // self.factor][j // self.factor]
            p2 = self.gray[i // self.factor + axis // 2][j // self.factor + axis % 2]
            return p1 + int((float(p2) - float(p1)) / self.factor * a)
        else:
            return self.gray[i // self.factor][j // self.factor]

    def main(self):
        array = numpy.zeros((len(self.gray) * self.factor, len(self.gray[0]) * self.factor), dtype=numpy.uint8)

        for i in range(0, len(array), self.factor):
            for j in range(len(array[0])):
                array[i][j] = self.inter(i, j, 1)

        for i in range(len(array)):
            for j in range(len(array[0])):
                if array[i][j] == 0:
                    array[i][j] = self.inter(i, j, 2)
        dst = cv2.merge([array])
        cv2.imshow("src", self.gray)
        cv2.imshow("Linear", dst)
        cv2.waitKey(30000)


if __name__ == '__main__':
    nearest = Nearest()
    nearest.main()
    linear = Linear()
    linear.main()
