#include "Conv.h"
#include <RunMgr.h>

namespace PyTZone {
namespace core {
// =========================================================
// void ConvNd::setDims(int64_t dims[MAX_CONV_DIMENSIONS]) {
//     memcpy(_dims, dims, MAX_CONV_DIMENSIONS * sizeof(int64_t));
// }
// void ConvNd::setOutDims(int64_t out_dims[MAX_CONV_DIMENSIONS]) {
//     memcpy(_out_dims, out_dims, MAX_CONV_DIMENSIONS * sizeof(int64_t));
// }
// =========================================================
// =========================================================
// =================================================== ConvNd
ConvNd::~ConvNd() {
    LogDebug("~ConvNd()");
}
#if SERIALIZER
ConvNd::ConvNd(ConvOpsCtx &coc):
    // 数据结构已近改变 SERIALIZER 目前被弃用
    // _dims{},
    // _out_dims{},
    // _binary(coc.get_binary()),
    // _xnor(coc.get_xnor()),
    _channel(coc.get_channel()),
    _num(coc.get_num()),
    // 这里摸钱也是直接深拷贝
    _size(coc.get_size()),
    _stride(coc.get_stride()),
    _padding(coc.get_padding()),
    _dilation(coc.get_dilation()),
    _groups(coc.get_groups()),
    _isBias(coc.get_isBias()),
    _padding_mode(static_cast<PADDING_MODE>(int64ToSizeT(coc.get_padding_mode()))),
    _dimsNum(static_cast<DIMENSIONALITY>(int64ToSizeT(coc.get_dimsNum())))
{
    LogDebug("ConvNd(ConvOpsCtx &)");
    _size.truncateArr(_dimsNum - 1);
    _stride.truncateArr(_dimsNum - 1);
    _padding.truncateArr(_dimsNum - 1);
    _dilation.truncateArr(_dimsNum - 1);
    setDimsV(coc.get_dims(), coc.get_out_dims(), _dims, _out_dims, _dimsNum);
}
#endif // SERIALIZER
#if LIBTORCH_EXT
ConvNd::ConvNd(SerConvNdPrePack &&spp):
    _channel(std::get<1>(spp)),
    _num(std::get<2>(spp)),
    // 这里目前也是直接深拷贝
    _size(std::move(std::get<3>(spp))),
    _stride(std::move(std::get<4>(spp))),
    _padding(std::move(std::get<5>(spp))),
    _dilation(std::move(std::get<6>(spp))),
    _groups(std::get<7>(spp)),
    _isBias(std::get<8>(spp)),
    _padding_mode(static_cast<PADDING_MODE>(int64ToSizeT(std::get<9>(spp))))
{
    LogDebug("ConvNd(SerConvNdPrePack &&)");
    _size.truncateArr(_dimsNum - 1);
    _stride.truncateArr(_dimsNum - 1);
    _padding.truncateArr(_dimsNum - 1);
    _dilation.truncateArr(_dimsNum - 1);
}
#endif // LIBTORCH_EXT
ConvNd::ConvNd(int64_t channel, int64_t num,
               const vector<int64_t> &size,  // kernel_size 不设置默认值  
               const vector<int64_t> &stride,
               const vector<int64_t> &padding,
               /* dilation: _size_2_t = 1, */
               const vector<int64_t> &dilation,
               int64_t groups,
               int8_t isBias,
               PADDING_MODE padding_mode
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
    _size.truncateArr(_dimsNum - 1);
    _stride.truncateArr(_dimsNum - 1);
    _padding.truncateArr(_dimsNum - 1);
    _dilation.truncateArr(_dimsNum - 1);
    // _idx = NETWORK->attachLayer(this);
    // EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

ConvNd::ConvNd(int64_t channel, int64_t num,
               const vector<int64_t> &size,  // kernel_size 不设置默认值  
               const vector<int64_t> &stride,
               const vector<int64_t> &padding,
               /* dilation: _size_2_t = 1, */
               const vector<int64_t> &dilation,
               int64_t groups,
               int8_t isBias,
               const string padding_mode
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
    _padding_mode(stringToPM(padding_mode))
{
    LogDebug("ConvNd(...)");
    _size.truncateArr(_dimsNum - 1);
    _stride.truncateArr(_dimsNum - 1);
    _padding.truncateArr(_dimsNum - 1);
    _dilation.truncateArr(_dimsNum - 1);

    _nweights = getKernelSize() * _channel * _num / _groups;
    if (_isBias) _nbiases = _num;
}

// template<typename... StrideArgs, typename... PaddingArgs, typename... DilationArgs>
// Conv2d::Conv2d(int64_t channel, int64_t num,
//                const vector<int64_t> &size,  // kernel_size 不设置默认值  
//                const tuple<StrideArgs...> &stride,
//                const tuple<PaddingArgs...> &padding,
//                /* dilation: _size_2_t = 1, */
//                const tuple<DilationArgs...> &dilation,
//                int8_t groups,
//                int8_t isBias,
//                PADDING_MODE padding_mode
//                /* device */
//                /* dtype */
//                ): 
ConvNd::ConvNd(int64_t channel, int64_t num,
               const vector<int64_t> &size,  // kernel_size 不设置默认值  
               const tuple<int64_t, int64_t> &stride,
               const tuple<int64_t, int64_t> &padding,
               /* dilation: _size_2_t = 1, */
               const tuple<int64_t, int64_t> &dilation,
               int64_t groups,
               int8_t isBias,
               PADDING_MODE padding_mode
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
    _size.truncateArr(_dimsNum - 1);
    _stride.truncateArr(_dimsNum - 1);
    _padding.truncateArr(_dimsNum - 1);
    _dilation.truncateArr(_dimsNum - 1);
// #ifdef DEBUGFLAG
//     std::cout << "\nStride size: " << std::tuple_size<decltype(stride)>::value << std::endl;    
//     std::cout << "Padding size: " << std::tuple_size<decltype(padding)>::value << std::endl;
//     std::cout << "Dilation size: " << std::tuple_size<decltype(dilation)>::value << std::endl;
// #endif
    // _idx = NETWORK->attachLayer(this);
    // EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

// ====================================================
#if LIBTORCH_EXT
void ConvNd::initialize(const at::Tensor &data) {
    LogDebug("ConvNd::initialize(const at::Tensor &)");
    // EXIT_ERROR_CHECK(data.dtype() == at::kFloat, false, "type in data if not FLOATCA");
    // EXIT_ERROR_CHECK(_inputs, SIZE_MAX, "_inputs error");
    int64_t dims_tmp[MAX_CONV_DIMENSIONS] = {0};
    get_tensor_dimensions(data, dims_tmp);
    _batch = getDimsIdx(dims_tmp, 1);
    // _inputs 原则上将应该用 network 中获取 但是会出现断层的情况
    _inputs = static_cast<int64_t>(data.numel()) / _batch;
    // 假定对于二维卷积来说 输入都是三维的
    EXIT_ERROR_CHECK(dims_tmp[2] == _channel, false, "channel for inputs error");
    setDims(_dims, dims_tmp, SizeTToInt64(static_cast<size_t>(_dimsNum)));     // _dims
    setOutDims(_dims, _out_dims, _num, 
               _size.getVector(), 
               _stride.getVector(),
               _padding.getVector());      // _out_dims
    Layer::setOutputs(_out_dims); 
    
    // l.output = calloc(l.batch*l.outputs, FLOAT_SIZE);
    // l.delta  = calloc(l.batch*l.outputs, FLOAT_SIZE);
    // int64_t _workspace_szie; 
    Layer::setWorkspaceSize(_dims, _out_dims, _groups, _size.getVector(), _dimsNum);
    // 更新 inputs outputs inoutSize for Net
    updateNet();
    // int8_t _binary, _xnor; 
}
#endif // LIBTORCH_EXT

void ConvNd::set_conv_out_height() {
    // return (l.h + 2*l.pad - l.size) / l.stride + 1;
    _out_dims[2] = (getHigh(_dims) + 2 * _padding.getData(0) - _size.getData(0)) / _stride.getData(0) + 1;
}
void ConvNd::set_conv_out_width() {
    // return (l.w + 2*l.pad - l.size) / l.stride + 1;
    _out_dims[3] = (getWeight(_dims) + 2 * _padding.getData(1) - _size.getData(1)) / _stride.getData(1) + 1;
}
#if LIBTORCH_EXT
const SerConvNdPrePack &ConvNd::makePack() {
    LogDebug("makePack() for conv");
        // 这里使用移动语义转移 Tensor 数据后 
        // 原来的 ops : 
        // weights ...    : [ Tensor (undefined) ] 
        // biases ...     : [ Tensor (undefined) ] 
        // 至于其他的 tuple 或是 vector 倒是无所吊谓
    SerConvNdPrePack *pspp = new SerConvNdPrePack(
        std::move(makeLPack()),
        // 以下的虽然调用 std::move 但是参数原本就是临时变量
        _channel,
        _num,
        std::move(_size.getVector()),
        std::move(_stride.getVector()),
        std::move(_padding.getVector()),
        std::move(_dilation.getVector()),
        _groups,
        _isBias,
        static_cast<int64_t>(_padding_mode));
    return *pspp;
}

void ConvNd::paramLoad(const string &str, const at::Tensor &data) {
    LogDebug("ConvNd::Load_param(const std::string &, const at::Tensor &)");
    paramLoad_for_datas(str, data);
    if (!strcmp(str.c_str(), "inputs")) {
        initialize(data);
    }
    // 这里的初始化函数都是以 layer 为单位来设置 
    // 关于 network 的同步化问题暂且不做考虑
}

#endif // LIBTORCH_EXT
#if TORCHZONE
void ConvNd::make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION) const {
    LogDebug("make_layer_ext_CA(TEEC_INVITATION_T * = nullptr) : idx = %d", _idx);
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);
    // 这里的设计就比较无奈了 TA 接口对接没有做好
    // 所幸的是我们并不关注初始化过程的效率问题
    INT_TA size[MAX_CONV_DIMENSIONS];
    INT_TA stride[MAX_CONV_DIMENSIONS];
    INT_TA padding[MAX_CONV_DIMENSIONS];
    INT_TA dilation[MAX_CONV_DIMENSIONS];

    for (size_t i = 0; i < MAX_CONV_DIMENSIONS; ++i) {
        size[i] = Int64TToINTTA(_size.getData(i));
        stride[i] = Int64TToINTTA(_stride.getData(i));
        padding[i] = Int64TToINTTA(_padding.getData(i));
        dilation[i] = Int64TToINTTA(_dilation.getData(i));
    }

    make_conv_ca(invitation,
                 static_cast<uint32_t>(_type),
                 _idx,
                 Int64TToINTTA(_channel),
                 Int64TToINTTA(_num),
                 size,
                 stride,
                 padding,
                 dilation,
                 Int64TToINTTA(_groups),
                 _isBias,
                 static_cast<uint32_t>(_padding_mode));
}
#endif

// bool Conv2d::checkStr(const string &str) const {
//     // 暂时 Layer 中没有初始化并保存 name 
//     // 所以检查逻辑比较简单
//     // 取决于 网络定义时的 变量名 格式 conv + idx + _ca
//     return string(string("conv") + to_string(_idx) + string("_ca")) == str;
// }

void ConvNd::print() const {
    cout << "!!!!!!!!!!!!!! print() :  ConvNd !!!!!!!!!!!!!!" << endl \
         << "idx            : " << (getIdx()) << endl \
         << "type           : " << _type << endl \
         << "keepIn         : " << ((_keepIn) ? "ture" : "false") << endl \
         << "keepOut        : " << ((_keepOut) ? "ture" : "false") << endl \
         << "batch          : " << _batch << endl \
         << "inputs         : " << _inputs << endl \
         << "outputs        : " << _outputs << endl \
         << "nweights       : " << _nweights << endl \
         << "nbiases        : " << _nbiases << endl \
         /* << "weights ...    : " << *_weights << "   " << *(_weights + 1) << "   " << *(_weights + _nweights - 2) << "   " << *(_weights + _nweights - 1) << endl \ */
         /* << "biases ...     : " << *_biases << "   " << *(_biases + 1) << "   " << *(_biases + _nbiases - 2) << "   " << *(_biases + _nbiases - 1) << endl \ */
         /* << "weights ...    : " << *_weights << "   " << *(_weights + 1) << "   " << *(_weights + _nweights - 2) << "   " << *(_weights + _nweights - 1) << endl \ */
         /* << "biases ...     : " << *_biases << "   " << *(_biases + 1) << "   " << *(_biases + _nbiases - 2) << "   " << *(_biases + _nbiases - 1) << endl \ */
         /* << "weights ...    : " << _weights << endl \ */
         /* << "biases ...     : " << _biases << endl \ */
         << "workspace_size : " << _workspace_size << endl;

    // // 固定格式设置为小数点后两位
    // std::cout << std::fixed << std::setprecision(4); 
    cout << "==========================================" << endl;
    cout << "weights ..." << endl;
    if (nullptr != _weights) {        
        for (int64_t i = 0; i < _nweights && i < 40; ++i) {
            if (!(i % 10) && 0 != i) cout << endl;
            printf("%+.4f  ", _weights[i]); // "+":始终显示符号
        }
        cout << endl;
    }
    cout << "==========================================" << endl;
    cout << "biases ..." << endl;
    if (_isBias && nullptr != _biases) {
        for (int64_t i = 0; i < _nbiases && i < 40; ++i) {
            if (!(i % 10) && 0 != i) cout << endl;
            printf("%+.4f  ", _biases[i]);
        }
    }
    cout << endl;
    cout << "==========================================" << endl;

    cout << "dims           : ";
    for (int64_t i = 1; i <= _dims[0]; ++i) cout << _dims[i] << " ";
    cout << endl;

    cout << "out_dims       : ";
    for (int64_t i = 1; i <= _out_dims[0]; ++i) cout << _out_dims[i] << " ";
    cout << endl;

    cout << "binary         : " << (_binary ? "true" : "false") << endl \
         << "xnor           : " << (_xnor ? "true" : "false") << endl;

    cout << "channel        : " << _channel << endl \
         << "num            : " << _num << endl;

    cout << "size           : ";
    _size.print();
    cout << "stride         : ";
    _stride.print();
    cout << "padding        : ";
    _padding.print();
    cout << "dilation       : ";
    _dilation.print();
    
    cout << "groups         : " << _groups << endl \
         << "isBias         : " << (_isBias ? "true" : "false") << endl \
         << "padding_mode   : " << _padding_mode << endl \
         << "dimsNum        : " << _dimsNum << endl;
    return;
}
void ConvNd::memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const {
    heapAll += _outputs;
    if (_keepIn) heapApply += _inputs;
    if (_keepOut) heapApply += _outputs;

    heapAll += _nweights;
    if (_isBias) heapAll += _nbiases;
    heapApply += _nweights;
    if (_isBias) heapApply += _nbiases;
    heapWeightsOnly += _nweights;
    if (_isBias) heapWeightsOnly += _nbiases;
}

// ===================================================
#if SERIALIZER
Conv2d::Conv2d(ConvOpsCtx &coc):
    Layer(coc),
    ConvNd(coc)
{
    LogDebug("Conv2d(ConvOpsCtx &)");
    int32_t ret = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(ret == _idx, false, "error idx from ConvOpsCtx");
    updateNet();
}
#endif // SERIALIZER
#if LIBTORCH_EXT
Conv2d::Conv2d(SerConvNdPrePack spp):
    Layer(std::move(std::get<0>(spp))),
    ConvNd(std::move(spp))
{
    LogDebug("Conv2d(SerConvNdPrePack)");
    int32_t ret = NETWORK->attachLayer(this, _idx);
    // cout << "ret == " << ret << endl;
    // EXIT_ERROR_CHECK(ret == _idx, false, "error idx from OpsCtx");
    updateNet();
}
#endif // LIBTORCH_EXT

Conv2d::Conv2d(int64_t channel, int64_t num,
               const vector<int64_t> &size,  // kernel_size 不设置默认值  
               const vector<int64_t> &stride,
               const vector<int64_t> &padding,
               /* dilation: _size_2_t = 1, */
               const vector<int64_t> &dilation,
               int64_t groups,
               int8_t isBias,
               PADDING_MODE padding_mode
               /* device */
               /* dtype */
               ):
    Layer(CONV_TYPE, D2),
    ConvNd(channel, num, size, stride, padding, dilation, groups, isBias, padding_mode)
{
    LogDebug("Conv2d(...)");
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

Conv2d::Conv2d(int64_t channel, int64_t num,
               const vector<int64_t> &size,  // kernel_size 不设置默认值  
               const vector<int64_t> &stride,
               const vector<int64_t> &padding,
               /* dilation: _size_2_t = 1, */
               const vector<int64_t> &dilation,
               int64_t groups,
               int8_t isBias,
               const string padding_mode
               /* device */
               /* dtype */
               ):
    Layer(CONV_TYPE, D2),
    ConvNd(channel, num, size, stride, padding, dilation, groups, isBias, padding_mode)
{
    LogDebug("Conv2d(...)");
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

// template<typename... StrideArgs, typename... PaddingArgs, typename... DilationArgs>
// Conv2d::Conv2d(int64_t channel, int64_t num,
//                const vector<int64_t> &size,  // kernel_size 不设置默认值  
//                const tuple<StrideArgs...> &stride,
//                const tuple<PaddingArgs...> &padding,
//                /* dilation: _size_2_t = 1, */
//                const tuple<DilationArgs...> &dilation,
//                int8_t groups,
//                int8_t isBias,
//                PADDING_MODE padding_mode
//                /* device */
//                /* dtype */
//                ): 
Conv2d::Conv2d(int64_t channel, int64_t num,
               const vector<int64_t> &size,  // kernel_size 不设置默认值  
               const tuple<int64_t, int64_t> &stride,
               const tuple<int64_t, int64_t> &padding,
               /* dilation: _size_2_t = 1, */
               const tuple<int64_t, int64_t> &dilation,
               int64_t groups,
               int8_t isBias,
               PADDING_MODE padding_mode
               /* device */
               /* dtype */
               ): 
    Layer(CONV_TYPE, D2),
    ConvNd(channel, num, size, stride, padding, dilation, groups, isBias, padding_mode)
{
    LogDebug("Conv2d(...)");
// #ifdef DEBUGFLAG
//     std::cout << "\nStride size: " << std::tuple_size<decltype(stride)>::value << std::endl;    
//     std::cout << "Padding size: " << std::tuple_size<decltype(padding)>::value << std::endl;
//     std::cout << "Dilation size: " << std::tuple_size<decltype(dilation)>::value << std::endl;
// #endif
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

Conv2d::Conv2d(const Conv2d &rhs):
    Layer(CONV_TYPE, D2),
    ConvNd(rhs._channel, rhs._num, 
    rhs._size,  rhs._stride,  rhs._padding,  rhs._dilation,  
    rhs._groups,  rhs._isBias,  rhs._padding_mode)
{
    LogDebug("Conv2d::Conv2d(const Conv2d &)");
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

// ==================================================================================


Conv2d::~Conv2d() {
    LogDebug("~Conv2d()");
}

#if LIBTORCH_EXT
void Conv2d::operator()(at::Tensor &input) {
    LogDebug("void Conv2d::operator()(at::Tensor &)");
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
}
void Conv2d::operator()() {
    LogDebug("void Conv2d::operator()()");
    NETWORK->registerIdx(_idx);
}

void Conv2d::attach(at::Tensor &input) {
    LogDebug("void Conv2d::attach(at::Tensor &) : idx = %d : ", _idx);
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
    // return this;
}
void Conv2d::attach() {
    LogDebug("void Conv2d::attach() : idx = %d : ", _idx);
    NETWORK->registerIdx(_idx);
    // return this;
}
#endif // LIBTORCH_EXT

// ==================================================================================

} // namespace end of core
} // namespace end of PyTZone
