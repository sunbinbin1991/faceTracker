# Face Tracker
基于mtcnn人脸检测+人脸跟踪（光流跟踪），就是要快快快 稳稳稳

## 开发环境
ncnn Windows Vs2015 Win64

## 开源框架
+ [ncnn](https://github.com/Tencent/ncnn)

+ [opencv](https://github.com/opencv/opencv)

## 引用
[HyperFT](https://github.com/zeusees/HyperFT)：参考部分代码架构

[ZQCNN](https://github.com/zuoqing1988/ZQCNN):包含人脸识别系统常用组件模型库，并带有加速；

[ncnn_example](git@github.com:MirrorYuChen/ncnn_example.git)：包含部分模型转换文件

编译方步骤：

+ 1、修改CMakeList.txt中的opencv/ncnn路径，修改成你自己的路径；
+ 2、mkdir build
+ 3、cd build
+ 4、cmake .. -G "Visual Studio 14 2015 Win64"
+ 5、make -j4


## 注意事项

1：运行时依赖的动态库，需要拷贝到可行性文件同级目录，或是参考如下目录添加环境变量：https://blog.csdn.net/guyuealian/article/details/79412644

Todo：

1：跟踪模块完善；

