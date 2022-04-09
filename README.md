# Non-intrusive model order reduction(MOR) Data Driving methods

## TODO LIST

- [X] add logging feature
- [X] add visualize feature
- [X] add calculate relative error feature
- [ ] add PINN loss
- [ ] parameter tuning

## ENV

> win 11 64bit
>
> Python==3.9.7
>
> Torch==1.11.0
>
> For more python package versions, please see [requirements.txt](requirements.txt)

## GIT TIPS

```git
git init 
git push -u origin main # gitlab master github main
git pull 

git stash  # 将代码保存到暂存区
git stash list 
git stash show 
git stash apply
git rm -r env/ --cached
```

```shell
########### 配置python 环境  ##################
# 下载 Anaconda 镜像
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2019.03-Linux-x86_64.sh
# 安装 Anaconda
bash Anaconda3-2019.03-Linux-x86_64.sh

# 安装完后，将conda 添加到环境变量里
sudo vim ~/.bashrc
# 在文末添加 anaconda3/bin的位置
export PATH=$PATH:/fastone/users/u176xxxxxxx7/anaconda3/bin
# : wq 保存，并source 一次
source ~/.bashrc

# 创建虚拟环境
conda create -n env3.9 python=3.9
# 激活环境
source activate
conda activate env3.9
# 安装python库
pip install -r requirements.txt
# 关闭
conda deactivate


########### 将python 任务挂到后台 ##############
nohup python3 -u main.py > mian_out.out 2>&1 &
# 解释
# 末尾的“&”：表示后台运行程序
# “nohup” ：保证程序不被挂起
# “python3”：是执行 python 代码的命令 (此处指定 Python 版本为 Python3)
# “-u”：是为了禁止缓存，让结果可以直接进入日志文件 main_out.out（如果不加 - u，则会导致日志文件不会实时刷新代码中的 print 函数的信息）
# “Job.py”：是欲执行的 python 的源代码文件，此处为 main.py
# “Job_out.out”：是输出的日志文件
# “>”：是指将打印信息指定到日志文件
# “2>&1”：将标准错误输出转变化标准输出，可以将错误信息也输出到日志文件中（0-> stdin, 1->stdout, 2->stderr）

#查看后台任务
jobs
# 或者
ps -aux | grep python
```

## REF

### PYTORCH

[Pytorch Docs](https://pytorch.org/docs/stable/index.html)

> Pytorch 官方文档，写的很好，推荐！

[Pytorch 中文教程](https://pytorch.apachecn.org/#/README)

> 汉化的文档，质量还行，如果看不懂英文的可以看一下这个。

### ML

[[Video] Physics-Informed Neural Network(PINN): Algorithms , Applications and Software](https://www.bilibili.com/video/BV12P4y1V7sz?spm_id_from=333.337.search-card.all.click)

[[Video] 内嵌物理的深度学习](https://app6ca5octe2206.pc.xiaoe-tech.com/detail/v_61149143e4b054ed7c4d0b26/3?fromH5=true)

> 这两个个是陆路（LU LU）的 PINNS 讲座，推荐！

[[Video] A Hands-on Introduction to Physics-informed Machine Learning](https://www.youtube.com/watch?v=o9JaZGWekWQ&ab_channel=nanohubtechtalks)

> 从神经网络讲到PINNs, 内容还不错，但是音频质量太差了。

[[code] PINN code demon1](https://github.com/cwq2016/POD-PINN)

> 这里面有 PINN、PDNN的算法实现，并且测试用例有1、2、3维的数据。使用的是Pytorch进行实现的。

[[code] PINN_code_demon2](https://github.com/maziarraissi/PINNs)

> 这个是非常著名的框架，作者是Maziar Raissi，必看！

[[code] PINN_code_demon3](https://github.com/jayroxis/PINNs.git)

> 这个是使用Maziar Raissi的代码框架，然后作者自己实现了pytorch、tensorflow的代码。

[[code] PINN_Papers_Summarize_repo](https://github.com/idrl-lab/PINNpapers)

> 总结 PINNs papers 的repo。

[[doc] ML Glossary ---> RNN architectures](https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html)

> 主要讲RNN、Auto encoder、Gan等网络的架构，值得阅读，这个网站也不错，可以深度探索。
