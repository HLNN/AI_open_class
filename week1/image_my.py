import cv2


class Image:
    def __init__(self, src=None, path="lena.jpg"):
        if src is not None:
            self.src = image
        else:
            self.src = cv2.imread(path)
        self.srcGray = None
        self.srcCanny = None

    def gray(self):
        self.srcGray = self.src.copy()
        for i in range(len(self.src)):
            for j in range(len(self.src[0])):
                # Gray = 0.30R + 0.59G + 0.11B
                self.srcGray[i][j] = 0.11 * self.src[i][j][0] + 0.59 * self.src[i][j][1] + 0.30 * self.src[i][j][2]

    def canny(self):
        if self.srcGray is None:
            self.gray()
        self.srcCanny = self.srcGray.copy()
        for i in range(1, len(self.src) - 1):
            for j in range(1, len(self.src[0]) - 1):
                fx = -1 * self.srcGray[i - 1][j - 1] + 1 * self.srcGray[i - 1][j + 1] + \
                     -2 * self.srcGray[i][j - 1] + 1 * self.srcGray[i][j] + 2 * self.srcGray[i][j + 1] + \
                     -1 * self.srcGray[i + 1][j - 1] + 1 * self.srcGray[i + 1][j + 1]
                fy = -1 * self.srcGray[i - 1][j - 1] + -2 * self.srcGray[i - 1][j] + -1 * self.srcGray[i - 1][j + 1] + \
                     1 * self.srcGray[i][j] + \
                     1 * self.srcGray[i + 1][j - 1] + 2 * self.srcGray[i + 1][j] + 1 * self.srcGray[i + 1][j + 1]
                self.srcCanny[i][j] = (abs(fx) + abs(fy)) // 2

    def show(self):
        cv2.imshow("Gray", self.srcGray)
        cv2.imshow("Sobel", self.srcCanny)
        cv2.waitKey()

    def get(self):
        self.canny()
        return self.srcCanny

    def main(self):
        self.gray()
        self.canny()
        self.show()


if __name__ == '__main__':
    image = Image()
    image.main()
