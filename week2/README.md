# AI开放实验课第一周总结

-------------------


## 目标

 - 使用最近邻插值、双线性插值和双立方插值等算法实现图像的放大
 - 使用等间隔采样和局部均值法实现图像的缩小


-----------------


#### 写在前面

因为昨天写完的时候自动保存的有点问题，之前写的全都没了，重写一次就懒得仔细去写了。


-----------------


### 图像放大

图像放大算法要解决的关键问题是如何对新加入的像素点进行插值。选择不同的插值算法会直接影响放大图像的质量和算法处理时间。


**最临近点插值**

最临近插值法是插值算法中最简单也是速度最快的。将放大后的图像上的像素点对应到原始图像上，并用原始图像上离对应点距离最近的像素的指表示。

```python
array = numpy.zeros((len(self.gray) * self.factor, len(self.gray[0]) * self.factor), dtype=numpy.uint8)
for i in range(len(array)):
    for j in range(len(array[0])):
        array[i][j] = self.gray[int(i / self.factor)][int(j / self.factor)]
dst = cv2.merge([array])
```

放大的速度虽快，但马赛克的感觉比较明显。

![nearest](https://github.com/HLNN/AI_open_class/blob/master/week2/pic/nearest.png)


**双线性插值**

双线性插值法也是先将放大后的图像上的像素对应到原始图像上，但不再使用原始图像上单一的像素点的指来表示，而是使用原始图像上距离最近的四个点的指加权表示。

双线性中的线性指的就是新加入的点对于原有的点，他的变化是线性的，而双是指这种线性插值是针对二为平面的两个方向的。

我首先写了`inter`函数，计算插值。

```python
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
```

然后遍历整个放大图像。

```python
array = numpy.zeros((len(self.gray) * self.factor, len(self.gray[0]) * self.factor), dtype=numpy.uint8)

for i in range(0, len(array), self.factor):
    for j in range(len(array[0])):
        array[i][j] = self.inter(i, j, 1)

for i in range(len(array)):
    for j in range(len(array[0])):
        if array[i][j] == 0:
            array[i][j] = self.inter(i, j, 2)
dst = cv2.merge([array])
```

边缘还不是特别清楚，但比之前的最邻近插值好了很多，没有明显的马赛克的感觉了。

![linear](https://github.com/HLNN/AI_open_class/blob/master/week2/pic/linear.png)


---------------


### 图像缩小

**等间隔采样**

等间隔采样法的思想就是对原始图片等间隔的进行抽样。实现思路和图像放大算法中的最临近插值法很像。

```python
array = numpy.zeros((len(self.gray) // self.factor, len(self.gray[0]) // self.factor), dtype=numpy.uint8)
for i in range(len(array)):
    for j in range(len(array[0])):
        array[i][j] = self.gray[int(i * self.factor)][int(j * self.factor)]
dst = cv2.merge([array])
```

![inter](https://github.com/HLNN/AI_open_class/blob/master/week2/pic/inter.png)


**局部均值**

局部均值法的思想是把原始图像按照缩放因子进行切片，用切片之后每个网格的平均值来表示缩小后一个像素点的值。


```python
array = numpy.zeros((len(self.gray) // self.factor, len(self.gray[0]) // self.factor), dtype=numpy.uint8)
for i in range(len(array)):
    for j in range(len(array[0])):
        if (i + 1) * self.factor < len(self.gray) and (j + 1) * self.factor < len(self.gray[0]):
            a = self.gray[i * self.factor: (i + 1) * self.factor, j * self.factor: (j + 1) * self.factor]
            array[i][j] = int(numpy.mean(a))
        else:
            array[i][j] = self.gray[i * self.factor][j * self.factor]
dst = cv2.merge([array])
```

![average](https://github.com/HLNN/AI_open_class/blob/master/week2/pic/average.png)

