# Pytorch Function Notes



## [torch.nn.Module()](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class for all neural network modules. Your models should also subclass this class.

> 所有神经网络模块的基类！您的模型也应该继承这个类！



### [torch.nn.Modeule.forward(*input)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward)

Defines the computation performed at every call. Should be overridden by all subclasses.

> 每次计算的时候调用，子类必须重写

## [torch.nn.Squential(*arg)](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential)

A sequential container.Modules will be added to it in the order they are passed in the constructor. Alternatively, an OrderedDict of modules can be passed in. The forward() method of Sequential accepts any input and forwards it to the first module it contains. It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.

The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the Sequential applies to each of the modules it stores (which are each a registered submodule of the Sequential).

What’s the difference between a Sequential and a torch.nn.ModuleList? A ModuleList is exactly what it sounds like–a list for storing Module s! On the other hand, the layers in a Sequential are connected in a cascading way.

> 顺序容器，`Modules` 会以他们传入的顺序添加到容器中。当然，也可以传入一个`OrderedDict`。执行的顺序就是这个顺序。



## [torch.optim](https://pytorch.org/docs/master/optim.html)

`torch.optim is` a package implementing various optimization algorithms.

> 是一个包，里面实现了很多常用的优化函数。
