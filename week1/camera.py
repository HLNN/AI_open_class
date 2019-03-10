import cv2


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def main(self):
        while True:
            ret, frame = self.cap.read()
            cv2.imshow('Camera', cv2.Canny(frame, 100, 300))
            cv2.waitKey(1)

if __name__ == '__main__':
    camera = Camera()
    camera.main()