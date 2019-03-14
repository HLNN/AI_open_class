# AI开放实验课第一周总结

-------------------


## 目标

 - 学习python基础
 - 学习git的使用
 - 简单图像处理，实现彩色图像转灰度，边缘提取

-----------------


### 学习python基础

python基础相对简单，之间[官网](https://www.python.org/)下载安装好就行，注意要把python目录加入到环境变量，方便命令行调用。

---------------


### 学习git的使用

git作为最常用的分布式版本控制系统，我们主要要了解git的`查看`，`添加`，`提交`，`拉取`，`推送`等命令。

 - 查看：查看当前分支的状态

```sh
git status
```

 - 添加：将文件加入缓冲区，可以指定文件或者添加整个目录

```sh
git add .
```

 - 提交：提交缓冲区的文件，``之后可以输入本次提交的说明信息

```sh
git commit -m "info"
```

 - 拉取：拉取远程更新

```sh
git pull origin master
```

 - 推送：推送本地更新到远程仓库

```sh
git push origin master
```

---------------------


### 简单图像处理

**基于opencv库函数**


 - 读取图片：使用`imread`函数读取图片文件，但注意`imread()`的结果是以`BGR`的顺序储存。

```python
self.src = cv2.imread(path)
```

 - 彩色图像转灰度：使用`cvtColor()`函数变换颜色空间。

```python
self.srcGray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
```

![gray_opencv](https://github.com/HLNN/AI_open_class/blob/master/week1/pic/gray_opencv.png)

 - 边缘提取：使用`Canny()`函数提取边缘特征，后面的两个参数表示的是提取的时候的梯度范围。

```python
self.srcCanny = cv2.Canny(self.src, 100, 300)
```

![canny_opencv](https://github.com/HLNN/AI_open_class/blob/master/week1/pic/canny_opencv.png)

 - 图片展示：使用`imshow()`函数显示图片并延时。

```python
cv2.imshow("Gray", self.srcGray)
cv2.imshow("Canny", self.srcCanny)
cv2.waitKey()
```


**不使用opencv库函数**


 - 彩色图像转灰度：遍历数据矩阵，按照`Gray = 0.30R + 0.59G + 0.11B`的关系计算每个像素的灰度值。

```python
for i in range(len(self.src)):
    for j in range(len(self.src[0])):
        # Gray = 0.30R + 0.59G + 0.11B
        self.srcGray[i][j] = 0.11 * self.src[i][j][0] + 0.59 * self.src[i][j][1] + 0.30 * self.src[i][j][2]
```

![gray_my](https://github.com/HLNN/AI_open_class/blob/master/week1/pic/gray_my.png)

 - 边缘提取：使用`Canny()`函数能提取出较好的边缘特征，但实现Canny算法需要高斯模糊，计算差分得到图像的梯度信息，最后在通过两个阈值确定强制边界和延迟边界，计算较为复杂。所以仅实现`Sobel`算子，效果很差。

```python
for i in range(1, len(self.src) - 1):
    for j in range(1, len(self.src[0]) - 1):
        fx = -1 * self.srcGray[i - 1][j - 1] + 1 * self.srcGray[i - 1][j + 1] + \
             -2 * self.srcGray[i][j - 1] + 2 * self.srcGray[i][j + 1] + \
             -1 * self.srcGray[i + 1][j - 1] + 1 * self.srcGray[i + 1][j + 1]
        fy = -1 * self.srcGray[i - 1][j - 1] + -2 * self.srcGray[i - 1][j] + -1 * self.srcGray[i - 1][j + 1] + \
             1 * self.srcGray[i + 1][j - 1] + 2 * self.srcGray[i + 1][j] + 1 * self.srcGray[i + 1][j + 1]
        self.srcCanny[i][j] = fx + fy
```

![sobel_my](https://github.com/HLNN/AI_open_class/blob/master/week1/pic/sobel_my.png)


**运行速度比较**


我使用timeit模块来计算代码运算的时间

```python
import timeit
import image_opencv
import image_my

i_opencv = image_opencv.Image()
i_my = image_my.Image()

print("Time of opencv BGRtoGray is {}".format(timeit.timeit(stmt=i_opencv.gray, number=5000)))
print("Time of my BGRtoGray is {}".format(timeit.timeit(stmt=i_my.gray, number=1)))

print("Time of opencv Canny is {}".format(timeit.timeit(stmt=i_opencv.canny, number=1000)))
print("Time of my Sobel is {}".format(timeit.timeit(stmt=i_my.canny, number=1)))
```


得到的测试结果是

```python
Time of opencv BGRtoGray is 0.440152151632584
Time of my BGRtoGray is 0.46648690575161617
Time of opencv BGRtoGray is 1.156731063888437
Time of my BGRtoGray is 1.6367132344208426
```

这说明我自己写的灰度转换算法差不多比opencv官方慢了五千倍，自己写的Sobel比官方的Canny慢了差不多一千倍......

-------------------


### 摄像头数据读取


opencv提供了读取摄像头数据的函数,使用`VideoCapture()`定义一个摄像头对象

```python
self.cap = cv2.VideoCapture(0)
```

调用`cap.read()`读取一帧的画面，两个返回值分别是状态码和一个`Mat`数组

```python
ret, frame = self.cap.read()
```

然后再调用Canny就能得到含边界信息的图片了

![camera](https://github.com/HLNN/AI_open_class/blob/master/week1/pic/camera.png)


### 视频处理


视频的读取和摄像头读取基本一样，直接改之前的代码就可以了。

保存视频的时候遇到了不少的问题。一开始我参考网上的代码使用`VideoWriter`，但得到的文件都是空文件或者只有文件头的几kb的小文件。

一开始以为是我选择的编码格式的问题，就试了好多个Google上别人用的编码方式，也在[fourcc官网](http://www.fourcc.org/codecs.php)上找了几个感觉和mp4相关的编码方式，但都不行。

后来我Google到了`VideoWriter`还有一个[`isopen`函数](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#cv2.VideoWriter.isopen)。我用这个函数试了一下，返回`False`，说明我的`VideoWriter`根本就没有打开成功。在同一个页面又找到一个叫[`open`函数](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#cv2.VideoWriter.open)。加上这个之后就可以了。下面的两段代码分别是获取`VideoWriter`设置所需的参数和设置`VideoWriter`。但还是觉得后面的两个函数有一点重复，不知道是不是我的打开方式有问题。

```python
self.fps = int(self.in_cap.get(cv2.CAP_PROP_FPS))
self.length = int(self.in_cap.get(cv2.CAP_PROP_FRAME_COUNT))
self.fourcc = int(self.in_cap.get(cv2.CAP_PROP_FOURCC))
self.frameSize = (int(self.in_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
```

```python
out_cap = cv2.VideoWriter('out.mp4', self.length, self.fourcc, self.fps, self.frameSize)
out_cap.open('out.mp4', self.fourcc, self.fps, self.frameSize)
```


----------------


#### 写在最后

因为考虑到方便展示经过Canny处理前后的视频，所以写的一个简单的`merge`函数。可以使用命令行参数或者`input`输入两个视频的文件名，就可以将两个视频水平拼接成一个。目前我只试了处理两个长度和分辨率完全相同的`mp4`文件，其他情况下可能会有报错。

