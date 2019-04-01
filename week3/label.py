import cv2


class Label:
    def __init__(self):
        self.src = cv2.imread("boat7_0927.jpg")
        with open("boat7_0927.txt") as f:
            info = f.readline()
            infos = info.split()
            self.cla = infos[0]
            self.xMin = int(infos[4])
            self.yMin = int(infos[5])
            self.xMax = int(infos[6])
            self.yMax = int(infos[7])
            print(self.xMin, self.xMax, self.yMin, self.yMax)
            print(type(self.yMax))

    def main(self):
        cv2.rectangle(self.src, (self.xMin, self.yMin), (self.xMax, self.yMax), (0, 0, 0), 2)
        cv2.putText(self.src, self.cla, (self.xMin, self.yMin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Label", self.src)
        cv2.waitKey()


if __name__ == '__main__':
    label = Label()
    label.main()
