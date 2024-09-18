---
# 文章标题
title: CS149 Lab Assignment1
# 设置写作时间
date: 2024-09-18
# 一个页面可以有多个分类
category:
  - CS149_Lab
# 一个页面可以有多个标签
tag:
  - 公开课
  - 并行计算
  - Lab
# 此页面会在文章列表置顶
sticky: true
# 此页面会出现在文章收藏中
star: true
# 侧边栏的顺序
# 数字越小越靠前，支持非整数和负数，比如 -10 < -9.5 < 3.2, order 为 -10 的文章会最靠上。
# 个人偏好将非干货或随想短文的 order 设置在 -0.01 到 -0.99，将干货类长文的 order 设置在 -1 到负无穷。每次新增文章都会在上一篇的基础上递减 order 值。
order: -1.01
---

## Prog1_mandelbort_threads

### 环境配置

本人使用OS为`Ubuntu 22.04`, 还是建议使用Linux系统做Lab, 很多环境配置会方便一些.  

CS149_Asst1并不需要额外配置运行环境, 下载解压一下编译环境就好啦!  
下载包:  

```
    wget https://github.com/ispc/ispc/releases/download/v1.21.0/ispc-v1.21.0-linux.tar.gz
```  

解压包:  

```
    tar -xvf ispc-v1.21.0-linux.tar.gz
```

配置环境路径:  

```
    export PATH=$PATH:${HOME}/Downloads/ispc-v1.21.0-linux/bin
```  

环境配置完成后就可以clone repo到本地来开始lab了:  

```  
    git clone https://github.com/stanford-cs149/asst1.git
```  

### 任务分析

> Pro1的内容主要是为了让学生了解`std::thread`的并行机制和"多线程不一定高效率"的并发事实, 所以难度并不算大~~(这是我的事后诸葛亮)~~, 整体框架已经在源码中基本完成了.完成后可以通过`make` + `./mandelbort --<args>`检验正确与否.

task :  

- 创建`线程0`和`线程1`, 分别计算图像的上下两个部分, 即`将图像的不同空间交给不同线程`计算, 这被称为`空间分解(spatial decomposition)`.
- 扩展代码使其能够使用`2, 3, 4, 5, 6, 7, 8`个线程, 进行空间分解, 生成加速图, 假设加速是否与线程数线性相关并加以验证.
- 在`workerThreadStart()`的开头和结尾插入计时代码, 验证并解释task2中提出的猜想.
- 修改一开始的线程分配方式, 实现将两个图片都拉到`8线程时7-8倍加速比`的效果, 找到适应任何线程数的泛型分配方式(不需要线程之间进行响应和同步), 报告最后得出的8线程加速比.
- 使用`16个线程`运行改进后代码, 回答性能是否明显高于8线程并解释原因.

事实上task中给的提示还是比较明显的, 在`task1`中解释了空间分解的概念, 那么通过对图片本身的`上下多份分割`,就可以解决这个问题,要注意分割的时候会不会漏行.  

### 任务实现

我们将一开始就对任务给出多线程的解决方式, 并在后续针对数据结果决定是否要进行优化.  

首先我们可以根据阅读`mandelbrotSerial.cpp`中的源码, 得到mandelbrotSerial()函数事实上是用来计算`Mandelbrot`图像的, 可以简单分析一下`mandelbrotSerial()`函数的各个参数:  

```
    void mandelbrotSerial(
    float x0, float y0, float x1, float y1, // 复平面左上和右下两个点坐标
    int width, int height,                  // 图像宽度和高度
    int startRow, int numRows,              // 开始行和总计算行数
    int maxIterations,                      // 最大迭代次数
    int output[]);                          // 每个点的迭代次数
```  

不难发现只要我们给出`startRow`, `numRows`, 其余保持图像默认参数, 就可以完成计算了.  
所以可以给出函数`workerThreadStart(WorkerArgs * const args)`的代码:  

```
    size_t rows = args -> height / args -> numThreads;          // 确定要计算的行数
    if (args -> height % args -> numThreads) {                  // 如果该遇到整除要加一行避免遗漏
        rows++;
    }
    size_t startRow = args -> threadId * rows;                  // 确定开始行
    // 如果已经到最后部分不够切分, 直接处理最后部分
    rows = rows > args -> height - startRow ? args -> height - startRow : rows;
    // 调用mandelbrotSerial
    mandelbrotSerial(args -> x0, args -> y0, args -> x1, args -> y1, args -> width, 
                    args -> height, startRow, rows, args -> maxIterations, args -> output);
``` 