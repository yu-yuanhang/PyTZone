#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn

# torch.classes.load_library("./../../TExt/build_cmake/libPyTZone.so")
# print(torch.classes.loaded_libraries)

ca_modules = []
idx = [0] 

def is_custom_module(module):
    if hasattr(module, 'forward') and len(list(module.children())) >= 1:
        return True
    return False 

def collect_modules(module, ca_modules, prefix=''):
    for name, sub_module in module.named_children():
        full_name = f"{prefix}{name}"
        if isinstance(sub_module, nn.Flatten): continue
        if is_custom_module(sub_module):
            collect_modules(sub_module, ca_modules, full_name + '.')
        else:
            ca_modules.append((full_name, sub_module))

def collect_ca_modules(module, prefix=''):
    # ca_modules = []
    for name, sub_module in module.named_children():
        full_name = f"{prefix}{name}"
        if full_name.endswith('_ca'):
            # 如果是自定义模块，递归查找其子模块
            if is_custom_module(sub_module):
                if isinstance(sub_module, nn.Sequential):
                    # print("nn.Sequential")
                    collect_modules(sub_module, ca_modules, full_name + '.')
                else:  
                    # print("nn.Module")
                    collect_modules(sub_module, ca_modules, full_name + '.')
            else:
                ca_modules.append((full_name, sub_module))
    # return ca_modules

def get_non_ca_modules(model):
    non_ca_modules = {}
    for name, module in model.named_children():
        if not name.endswith('_ca'):
            non_ca_modules[name] = module
    return non_ca_modules      

def get_index_by_name(ca_modules, name):
    for index, (module_name, module) in enumerate(ca_modules):
        if module_name == name:
            return index
    return -1  # 如果未找到，返回 -1            
    
                      
def create_hook(name, module):
    # module_names = list(ca_modules.keys())
    # # 获取目标模块名称在列表中的索引
    # idx = module_names.index(name) if name in module_names else -1
    
    # if idx == -1:
    #     print(f"Error: 模块 '{name}' 不存在于模块列表中.")
    #     sys.exit(1)  # 退出程序并返回状态码 1

    def hook_function(module, input, output):
        # print(f"Hook called for {name}, input shape: {input[0].shape}, output shape: {output.shape}")
        weights = module.weight.data  if hasattr(module, 'weight') and module.weight is not None else None  # 获取权重
        bias = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None        # 获取偏置
        # 在这里进行初始化或其他操作
        # 这里需要调用 network 的接口
        # torch.ops.my_ops.warp_perspective
        idx = get_index_by_name(ca_modules, name)
        if weights is not None:
            # print(f"Weights shape: {weights.shape}")
            torch.ops.PyTZone.paramShift('weight', weights, idx)
        if bias is not None:
            # print(f"Bias shape: {bias.shape}")
            torch.ops.PyTZone.paramShift('bias', bias, idx)
        torch.ops.PyTZone.paramShift('inputs', input[0], idx)
            
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            running_mean = module.running_mean.data
            running_var = module.running_var.data
            
            # print(f"Running Mean shape: {running_mean.shape}")
            # print(f"Running Variance shape: {running_var.shape}")
            
            # 使用 paramShift 调用均值和方差
            torch.ops.PyTZone.paramShift('mean', running_mean, idx)
            torch.ops.PyTZone.paramShift('variance', running_var, idx)

    return hook_function

# 这里暂且只考虑最简单的模型的形式 不存在条件控制语句
# 由于 pytorch 动态图的架构设计导致对应子模块对于输出和输出不感知
# 理论上这种设计和常见的边缘推理框架的静态图在设计上存在冲突
# 后续可能需要设计关于通过 TrochScript 的计算图来获取构建静态图所需的信息的接口
def weightShift(target_model, source_model, input_data):

    # 理论上这里也需要对 network 进行初始化
    # 但是对于目前的推理需求 net 内结构比较简单
    # todo initial net
    
    torch.ops.PyTZone.partionLayer()
    # torch.ops.PyTZone.printNet()
    
    collect_ca_modules(source_model)

    for sub_name, sub_module in ca_modules:
        print(f"{sub_name}: {sub_module}")
        sub_module.register_forward_hook(create_hook(f"{sub_name}", sub_module))

    # 处理不是以 _ca 结尾的模块，转移参数
    target_modules = get_non_ca_modules(target_model)
    source_modules = get_non_ca_modules(source_model)
    if len(target_modules) == len(source_modules):
        for (name1, module1), (name2, module2) in zip(target_modules.items(), source_modules.items()):
            target_model._modules[name1] = module2

    # 执行目标模型的前向传播
    output = source_model(input_data)
    # torch.ops.PyTZone.printNet()
    return output

