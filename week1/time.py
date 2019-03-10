import timeit
import image_opencv
import image_my

i_opencv = image_opencv.Image()
i_my = image_my.Image()

print("Time of opencv BGRtoGray is {}".format(timeit.timeit(stmt=i_opencv.gray, number=5000)))
print("Time of my BGRtoGray is {}".format(timeit.timeit(stmt=i_my.gray, number=1)))

print("Time of opencv Canny is {}".format(timeit.timeit(stmt=i_opencv.canny, number=1000)))
print("Time of my Sobel is {}".format(timeit.timeit(stmt=i_my.canny, number=1)))