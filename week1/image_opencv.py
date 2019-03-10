import cv2


class Image:
    def __init__(self, src=None, path="lena.jpg"):
        if src is not None:
            self.src = src
        else:
            self.src = cv2.imread(path)
        self.srcGray = None
        self.srcCanny = None

    def gray(self):
        self.srcGray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

    def canny(self):
        self.srcCanny = cv2.Canny(self.src, 100, 300)

    def show(self):
        cv2.imshow("Gray", self.srcGray)
        cv2.imshow("Canny", self.srcCanny)
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
