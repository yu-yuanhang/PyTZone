!/bin/bash

 执行 pip install .
 pip install -e .  # 创建符号链接，代码修改实时生效

 # 检查是否成功安装
 if [ $? -eq 0 ]; then
     echo "安装成功，正在移动文件..."

     # 移动 dist/ 和 torch_ca.egg-info/ 到 build 目录下
     # mv dist build/
     # mv torch_ca.egg-info build/

     echo "文件移动完成！"
 else
     echo "安装失败，请检查错误信息。"
 fi
