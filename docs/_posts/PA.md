---
# 文章标题
title: PA -- A new journey for the OS
# 设置写作时间
date: 2024-12-26
# 一个页面可以有多个分类
category:
  - 计算机体系结构
# 一个页面可以有多个标签
tag:
  - 系统
  - 计算机组成
# 此页面会在文章列表置顶
sticky: true
# 此页面会出现在文章收藏中
star: true
# 侧边栏的顺序
# 数字越小越靠前，支持非整数和负数，比如 -10 < -9.5 < 3.2, order 为 -10 的文章会最靠上。
# 个人偏好将非干货或随想短文的 order 设置在 -0.01 到 -0.99，将干货类长文的 order 设置在 -1 到负无穷。每次新增文章都会在上一篇的基础上递减 order 值。
order: -3
--- 
> This blog will begin from the half of PA1
> Because I thought "STFW" can solve the problems before.
> I have to confirm that I have install the Ubuntu 22.04 for my CS task, so I look through the PA0 in a fast way.

## PA1 The simplest computer

### infrastructure

> There always exist infrastructure where exist codes.

The course build a significant infrainstructure called `Simple Debugger`(sdb) in the NEMU.  

NEMU is regarded as a programm to excute other guest programm which means NEMU can know all information of the guest programm.  

However, the information is hard to be caught by the debugger out of the NEMU, such as GDB.(set a breakpoint by GDB for the guest programm is also hard)  

Inorder to improve the efficiency of debugging, we need to build a simple Debugger at the monitor.

Below are the format and functions:  
|instruction|format|example|explanation|
|-----------|------|-------|-----------|
|help|`help`|`help`|print help of instructions|
|continue|`c`|`c`|continue to run the programm is suspended|
|quit|`q`|`q`|quit NEMU|
|run by step|`si [N]`|`si 10`|run N instructions by step then stop, default N is equal to 1|
|print program status|`info SUBCMD`|`info r`<br>`info w`| print register status <br> print monitor message|
|scale Memory| `x N EXPR` | `x 10 $esp` | figure out the value of EXPR and set it as the begining of the Memory. output N*4 bytes as hex format|
|figure out the expression value|`p EXPR`|`p $eax + 1`|figure out EXPR value|
|set monitor|`w EXPR`|`w *0x2000`|when EXPR changes, suspend the program|
|delete monitor|`d N`|`d 2`|delete monitor whose label is N|
