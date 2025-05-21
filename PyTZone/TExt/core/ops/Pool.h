#ifndef __POOL_H__
#define __POOL_H__

#include <core/Layer.h>
#include <core/network.h>
#include <random> 

namespace PyTZone {
namespace core {

// =============================================================maxpool
// kernel_size : 池化窗口的大小 类型为 _size_any_t 
//     这个类型可能是 int 或 tuple 表示池化窗口的大小 
// stride : 池化窗口的滑动步幅 默认为 None 
//     如果未提供 默认与 kernel_size 相同
// padding : 默认 0 不进行填充
// dilation : 池化操作的扩张系数 默认 1 表示没有扩张
// return_indices : 默认为 false 
//     如果为 true 则返回池化时的索引（作用于反向传播时恢复最大值位置）
// ceil_mode : 默认为 false
//     如果为 true 池化的输出尺寸将向上取整
// =============================================================
// maxpool_layer l = {0};
// l.type = MAXPOOL_TYPE;
// l.batch = batch;
// l.h = h;
// l.w = w;
// l.c = c;
// l.pad = padding;
// l.out_w = (w + padding - size)/stride + 1;
// l.out_h = (h + padding - size)/stride + 1;
// l.out_c = c;
// l.outputs = l.out_h * l.out_w * l.out_c;
// l.inputs = h*w*c;
// l.size = size;
// l.stride = stride;
// int output_size = l.out_h * l.out_w * l.out_c * batch;
// l.indexes = calloc(output_size, sizeof(int));
// l.output =  calloc(output_size, sizeof(float));
// l.delta =   calloc(output_size, sizeof(float));
// l.forward = forward_maxpool_layer;
// l.backward = backward_maxpool_layer;
// ===============================================================================
// =============================================================avgpool
// kernel_size
// stride  
// padding 
// ceil_mode
// count_include_pad:
//     True : 则填充的元素也会被计入计算平均值的总数
//     False(默认值) : 则只计算有效元素
// divisor_override: 默认 None

class PoolNd:
#if LIBTORCH_EXT
public torch::CustomClassHolder,
#endif // LIBTORCH_EXT
virtual public Layer
{    
public:
    void print() const override;
#if LIBTORCH_EXT
    const SerPoolPrePack &makePack(); 
    void paramLoad(const string &str, const at::Tensor &data) override;
    void initialize(const at::Tensor &data) override;
#endif // LIBTORCH_EXT
#if TORCHZONE
    void make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr) const override;
#endif
protected:
#if LIBTORCH_EXT
    PoolNd(SerPoolPrePack &&spp);
#endif // LIBTORCH_EXT
    PoolNd() = delete;
    ~PoolNd();
    PoolNd(const vector<int64_t> &size,  // kernel_size 不设置默认值  
              const vector<int64_t> &stride = vector<int64_t>(),
              const vector<int64_t> &padding = vector<int64_t>(),
              /* dilation: _size_2_t = 1, */
              const vector<int64_t> &dilation = vector<int64_t>(),
              int8_t return_indices = false,
              int8_t ceil_mode = false,
              int8_t count_include_pad = false,
              int64_t divisor_override = INVALID_VALUE_U,
              const string padding_mode = "ZEROS"
              /* device */
              /* dtype */);

protected:

    CustomArray<int64_t, MAX_CONV_DIMENSIONS> _size;
    // 这里先初始化为无效值 如果未提供后构造默认与 kernel_size 相同
    CustomArray_def<int64_t, MAX_CONV_DIMENSIONS, INVALID_VALUE_U> _stride;
    // 默认不进行填充
    CustomArray_def<int64_t, MAX_CONV_DIMENSIONS, 0> _padding;
    CustomArray_def<int64_t, MAX_CONV_DIMENSIONS, 1> _dilation;

    int8_t _return_indices; // 默认为 false 
    // 如果为 true 则返回池化时的索引（作用于反向传播时恢复最大值位置）
    int8_t _ceil_mode; // 默认为 false
    int8_t _count_include_pad;   // 默认为 false
    int64_t _divisor_override;   // 默认为 None
    PADDING_MODE _padding_mode;
};

class MaxPool2d:
public PoolNd {
public:
    MaxPool2d() = delete;
    ~MaxPool2d();
#if LIBTORCH_EXT
    MaxPool2d(SerPoolPrePack spp);
#endif // LIBTORCH_EXT

    MaxPool2d(const vector<int64_t> &size,  // kernel_size 不设置默认值  
              const vector<int64_t> &stride = vector<int64_t>(),
              const vector<int64_t> &padding = vector<int64_t>(),
              /* dilation: _size_2_t = 1, */
              const vector<int64_t> &dilation = vector<int64_t>(),
              int8_t return_indices = false,
              int8_t ceil_mode = false);

#if LIBTORCH_EXT
    void operator()(at::Tensor &input);
    void operator()();
    // 作用 同函数调用运算符
    void attach(at::Tensor &input);
    void attach();
#endif // LIBTORCH_EXT

    void forward() override;

};

class AvgPool2d:
public PoolNd {
public:
    AvgPool2d() = delete;
    ~AvgPool2d();
#if LIBTORCH_EXT
    AvgPool2d(SerPoolPrePack spp);
#endif // LIBTORCH_EXT

    // 底层算子没有替换 目前等价于 nn.AdaptiveAvgPool2d(1),
    AvgPool2d(const vector<int64_t> &size,  // kernel_size 不设置默认值  
              const vector<int64_t> &stride = vector<int64_t>(),
              const vector<int64_t> &padding = vector<int64_t>(),
              int8_t ceil_mode = false,
              int8_t count_include_pad = false,
              int64_t divisor_override = INVALID_VALUE_U);

#if LIBTORCH_EXT
    void operator()(at::Tensor &input);
    void operator()();
    // 作用 同函数调用运算符
    void attach(at::Tensor &input);
    void attach();
#endif // LIBTORCH_EXT

    void forward() override;

};


} // namespace end of core
} // namespace end of PyTZone

#endif