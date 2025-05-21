#ifndef __CONV_H__
#define __CONV_H__


#include <core/Layer.h>
#include <core/network.h>
#include <random> 

using std::array;
using std::tuple;
using std::cout;
using std::endl;
using std::string;

// =========================================================

// =========================================================
namespace PyTZone {
namespace core {

// =========================================================

class ConvNd: 
#if LIBTORCH_EXT
public torch::CustomClassHolder,
#endif // LIBTORCH_EXT
virtual public Layer
{
    // using Layer::makeParams;    // 重新声明用于序列化的函数 
#if SERIALIZER
    using Layer::getWeights;
    using Layer::getBiases;
    friend class ConvOpsCtx;
#endif  // SERIALIZER
// =========================================================
// =========================================================
public:
    inline int64_t getKernelSize() const {
        int64_t sz = 1;
        for (size_t i = 0; i < _size.getn(); ++i) sz *= _size.getData(i);
        return sz;
    }

    int64_t getNum_ConvNd() const override {return _num;}
    void updateNet_thr(List<struct subNet_s> &subNet, size_t threadSize) override;
    void print() const override;
    // void forward() const = 0;
    void memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const;

#if LIBTORCH_EXT
    // 这里用来序列化的函数 但是当前调用的还是深度拷贝
    // 后续关于 opsCtx 应用移动语义的序列化过程还待设计
    // (说白了 水平有限 工期紧迫 求放过) 反正目前只要不影响推理时效率就无所吊谓
    const SerConvNdPrePack &makePack();
    void paramLoad(const string &str, const at::Tensor &data) override;
    void initialize(const at::Tensor &data) override;

#endif // LIBTORCH_EXT
#if TORCHZONE
    void make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr) const override;
#endif

// protected:
// 为了确保序列化模块中的通用性需要 ConvNd 作为参数
protected:
    ConvNd() = delete;
    // ConvNd(): Layer() {};
#if SERIALIZER
    ConvNd(ConvOpsCtx &coc);
#endif  // SERIALIZER
#if LIBTORCH_EXT
    ConvNd(SerConvNdPrePack &&spp);
#endif // LIBTORCH_EXT
    ConvNd(int64_t channel, int64_t num,
           const vector<int64_t> &size,  // kernel_size 不设置默认值  
           const vector<int64_t> &stride = vector<int64_t>(),
           const vector<int64_t> &padding = vector<int64_t>(),
           /* dilation: _size_2_t = 1, */
           const vector<int64_t> &dilation = vector<int64_t>(),
           int64_t groups = 1,
           int8_t isBias = true,
           PADDING_MODE padding_mode = ZEROS_PADDING
           /* device */
           /* dtype */);
    ConvNd(int64_t channel, int64_t num,
           const vector<int64_t> &size,  // kernel_size 不设置默认值  
           const vector<int64_t> &stride = vector<int64_t>(),
           const vector<int64_t> &padding = vector<int64_t>(),
           /* dilation: _size_2_t = 1, */
           const vector<int64_t> &dilation = vector<int64_t>(),
           int64_t groups = 1,
           int8_t isBias = true,
           // 这里 枚举类型 暂时无法传递给 torchscript 导致
           // 传递给 torch::class_<Conv2d>::def 的参数类型仍然不匹配
           // 用 string 代替 PADDING_MODE
           // PADDING_MODE padding_mode = ZEROS_PADDING
           const string padding_mode = "ZEROS"
           /* device */
           /* dtype */);
    ConvNd(int64_t channel, int64_t num,
           const vector<int64_t> &size,  // kernel_size 不设置默认值  
           const tuple<int64_t, int64_t> &stride = std::tuple<int64_t, int64_t>(1, 1),
           const tuple<int64_t, int64_t> &padding = std::tuple<int64_t, int64_t>(1, 1),
           /* dilation: _size_2_t = 1, */
           const tuple<int64_t, int64_t> &dilation = std::tuple<int64_t, int64_t>(1, 1),
           int64_t groups = 1,
           int8_t isBias = true,
           PADDING_MODE padding_mode = ZEROS_PADDING
           /* device */
           /* dtype */);
    template <typename T, std::size_t N, T DefVal>
    ConvNd(int64_t channel, int64_t num,
           const CustomArray<T, N> &size,  // kernel_size 不设置默认值  
           const CustomArray_def<T, N, DefVal> &stride,
           const CustomArray_def<T, N, DefVal> &padding,
           /* dilation: _size_2_t = 1, */
           const CustomArray_def<T, N, DefVal> &dilation,
           int64_t groups = 1,
           int8_t isBias = true,
           PADDING_MODE padding_mode = ZEROS_PADDING
           /* device */
           /* dtype */):
        _channel(channel),
        _num(num),
        _size(size),
        _stride(stride),
        _padding(padding),
        _dilation(dilation),
        _groups(groups),
        _isBias(isBias),
        _padding_mode(padding_mode)
    {
        LogDebug("ConvNd(...)");
    }

    // virtual 虚析构
    ~ConvNd();
    // friend void pybind11_Conv2d(pybind11::module_ &);
protected:
    void set_conv_out_height();
    void set_conv_out_width();
    
protected:
    // ========================= 以下初始化交给子类
    int64_t _channel;
    int64_t _num;    // 卷积核的数量

    // 卷积核维度 Cin Kh Kw ...
    CustomArray<int64_t, MAX_CONV_DIMENSIONS> _size;   // 卷积核大小
    CustomArray_def<int64_t, MAX_CONV_DIMENSIONS, 1> _stride;
    // Union[str, _size_2_t]
    CustomArray_def<int64_t, MAX_CONV_DIMENSIONS, 1> _padding;
    /* dilation 空洞卷积的扩张率 定义卷积核元素之间的间隔 默认为 1 表示标准卷积 */
    // 这个变量当前还没有起作用
    CustomArray_def<int64_t, MAX_CONV_DIMENSIONS, 1> _dilation;
    int64_t _groups;
    int8_t _isBias;
    /* padding_mode 卷积填充模式 */
    PADDING_MODE _padding_mode;
    /* device */
    /* dtype 见的类型包括 torch.float32 和 torch.float64 */
};


class Conv2d: 
public ConvNd {
// public torch::CustomClassHolder {

public:
    Conv2d() = delete;
    // Conv2d(): Layer(), ConvNd() {};
#if SERIALIZER
    Conv2d(ConvOpsCtx &coc);
#endif  // SERIALIZER

#if LIBTORCH_EXT
    Conv2d(SerConvNdPrePack spp);
#endif // LIBTORCH_EXT

    Conv2d(int64_t channel, int64_t num,
           const vector<int64_t> &size,  // kernel_size 不设置默认值  
           const vector<int64_t> &stride = vector<int64_t>(),
           const vector<int64_t> &padding = vector<int64_t>(),
           /* dilation: _size_2_t = 1, */
           const vector<int64_t> &dilation = vector<int64_t>(),
           int64_t groups = 1,
           int8_t isBias = true,
           PADDING_MODE padding_mode = ZEROS_PADDING
           /* device */
           /* dtype */);
    // ================================================================== 
    // 暂时只用这个构造函数
    Conv2d(int64_t channel, int64_t num,
           const vector<int64_t> &size,  // kernel_size 不设置默认值  
           const vector<int64_t> &stride = vector<int64_t>(),
           const vector<int64_t> &padding = vector<int64_t>(),
           /* dilation: _size_2_t = 1, */
           const vector<int64_t> &dilation = vector<int64_t>(),
           int64_t groups = 1,
           int8_t isBias = true,
           // 这里 枚举类型 暂时无法传递给 torchscript 导致
           // 传递给 torch::class_<Conv2d>::def 的参数类型仍然不匹配
           // 用 string 代替 PADDING_MODE
           // PADDING_MODE padding_mode = ZEROS_PADDING
           const string padding_mode = "ZEROS"
           /* device */
           /* dtype */);
    // ==================================================================
    // template<typename... StrideArgs, typename... PaddingArgs, typename... DilationArgs>
    // Conv2d(int64_t channel, int64_t num,
    //        const vector<int64_t> &size,  // kernel_size 不设置默认值  
    //        const tuple<StrideArgs...> &stride = std::make_tuple(static_cast<int64_t>(1)),
    //        const tuple<PaddingArgs...> &padding = std::make_tuple(static_cast<int64_t>(0)),
    //        /* dilation: _size_2_t = 1, */
    //        const tuple<DilationArgs...> &dilation = std::make_tuple(static_cast<int64_t>(1)),
    //        int8_t groups = 1,
    //        int8_t isBias = true,
    //        PADDING_MODE padding_mode = ZEROS_PADDING
    //        /* device */
    //        /* dtype */);
    Conv2d(int64_t channel, int64_t num,
           const vector<int64_t> &size,  // kernel_size 不设置默认值  
           const tuple<int64_t, int64_t> &stride = std::tuple<int64_t, int64_t>(1, 1),
           const tuple<int64_t, int64_t> &padding = std::tuple<int64_t, int64_t>(1, 1),
           /* dilation: _size_2_t = 1, */
           const tuple<int64_t, int64_t> &dilation = std::tuple<int64_t, int64_t>(1, 1),
           int64_t groups = 1,
           int8_t isBias = true,
           PADDING_MODE padding_mode = ZEROS_PADDING
           /* device */
           /* dtype */);

    // 目前在拷贝构造函数中 还没有添加 权重 输入输出 等结构信息
    // 因为目前还没有拷贝构造函数的适用场景
    // 原本是设计来支持序列化和反序列化模块的
    // 但是处于架构设计的考虑目前已经弃用了
    // 在 torchScript 中给出的反序列化接口中自定义的智能指针 intrusive_ptr 类似于 auto_ptr
    // 在函数中返回 ret 并不调用拷贝构造函数 这里猜测数据是被独享的
    Conv2d(const Conv2d &rhs);

    ~Conv2d();

public:
    inline void setKeep(int8_t keepIn, int8_t keepOut) {
        LogDebug("void Conv2d::setKeep(int8_t, int8_t) : idx = %d : ", _idx);
        _keepIn = keepIn;
        _keepOut = keepOut;
        // return this;
    }
    inline int64_t getIdx() const {return this->_idx;}

    // 重载 函数调用运算符 operator()
    // at::Tensor operator()(at::Tensor &input);
    // at::Tensor operator()();
#if LIBTORCH_EXT
    void operator()(at::Tensor &input);
    void operator()();
    // 作用 同函数调用运算符
    // void attach(at::Tensor &input);
    void attach(at::Tensor &input);
    void attach();
#endif // LIBTORCH_EXT

    void forward() override;

private:
    // bool checkStr(const string &str) const;
    // convolutional_out_width()

    /* virtual void func() override */
    // 虚函数接口

private:
    
};

} // namespace end of core
} // namespace end of PyTZone

#endif
