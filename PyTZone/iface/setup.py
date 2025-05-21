from setuptools import setup, find_packages

# python setup.py install --prefix=/your/custom/path
# python setup.py install --prefix=/home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension/iface/build

# 可以使用以下命令安装
# pip install /path/to/toch_ca-0.1-py3-none-any.whl
# pip install /path/to/toch_ca-0.1.tar.gz

# pip install --prefix=/your/custom/path /path/to/toch_ca-0.1-py3-none-any.whl
# pip install --prefix=/home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension/iface/build /home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension/iface/build/toch_ca-0.1-py3-none-any.whl

setup(
    name='torch_ca',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # 'torch',
    ],  # 这里可以添加依赖包
)
