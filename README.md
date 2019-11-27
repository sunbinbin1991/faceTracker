# Face Tracker
基于mtcnn人脸检测+人脸跟踪（光流跟踪），就是要快快快 稳稳稳

## 开发环境
ncnn Windows Vs2015 Win64

## 开源框架
+ [ncnn](https://github.com/Tencent/ncnn)

+ [opencv](https://github.com/opencv/opencv)

## 测试结果
![test](https://github.com/sunbinbin1991/faceTracker/blob/master/data/IU_mark.jpg?raw=true)

## 引用
[HyperFT](https://github.com/zeusees/HyperFT)：参考部分代码模块

[ZQCNN](https://github.com/zuoqing1988/ZQCNN):包含人脸识别系统常用组件模型库，并带有加速；

[ncnn_example](https://github.com/MirrorYuChen/ncnn_example)：包含部分模型转换文件

编译步骤：

+ 1、修改CMakeList.txt中的opencv/ncnn路径，修改成你自己的路径；
+ 2、mkdir build
+ 3、cd build
+ 4、cmake .. -G "Visual Studio 14 2015 Win64"
+ 5、make -j4


## 注意事项

1：运行时依赖的动态库，需要拷贝到可执行文件同级目录，或是参考如下目录添加环境变量：https://blog.csdn.net/guyuealian/article/details/79412644

2: 跟踪测试demo中提供同步和异步跟踪方案，同步是指检测放入主线程，有检出后进行跟踪匹配；异步是指，检测模块有单独子线程，会将检出结果同步给跟踪主线程，用于匹配跟踪；

Todo：
- [x] 跟踪模块完善；3/11/2019
- [x] 跟踪模块优化；24/11/2019
- [x] 匹配逻辑需要修改, 当前的匹配逻辑仅仅是单纯的iou计算；
- [x] 光流法的参数需要调整；
- [x] 

---

# Face Tracker
Based on mtcnn face detection + face tracking (optical flow tracking), you need to be fast and stable

## Development Environment
ncnn Windows Vs2015 Win64

## Open source framework
+ [ncnn] (https://github.com/Tencent/ncnn)

+ [opencv] (https://github.com/opencv/opencv)

## Test Results
! [test] (https://github.com/sunbinbin1991/faceTracker/blob/master/data/IU_mark.jpg?raw=true)

## Quote
[HyperFT] (https://github.com/zeusees/HyperFT): Reference part of the code module

[ZQCNN] (https://github.com/zuoqing1988/ZQCNN): Contains a common component model library for face recognition systems, with acceleration;

[ncnn_example] (https://github.com/MirrorYuChen/ncnn_example): Contains some model conversion files

Compilation steps:

+ 1. Modify the opencv / ncnn path in CMakeList.txt to your own path;
+ 2, mkdir build
+ 3, cd build
+ 4, cmake .. -G "Visual Studio 14 2015 Win64"
+ 5, make -j4

## Precautions

1: The dynamic library that the runtime depends on needs to be copied to the same directory as the executable file, or refer to the following directory to add environment variables: https://blog.csdn.net/guyuealian/article/details/79412644

2: The tracking test demo provides synchronous and asynchronous tracking solutions. Synchronization means that the detection is placed in the main thread and the tracking is matched after detection. Asynchronous means that the detection module has a separate sub-thread and will synchronize the detection result to the tracking master Threads for matching tracking;

Todo:
- [x] Perfect tracking module; 3/11/2019
- [x] Tracking module optimization; 24/11/2019
- [x] The matching logic needs to be modified. The current matching logic is just a simple iou calculation;
- [x] parameters of optical flow method need to be adjusted;
- [x]