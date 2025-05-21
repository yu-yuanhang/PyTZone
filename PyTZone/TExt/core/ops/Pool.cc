#include "Pool.h"
#include <RunMgr.h>

namespace PyTZone {
namespace core {

PoolNd::~PoolNd() {
    LogDebug("~PoolNd()");
    destory();
}

PoolNd::PoolNd(const vector<int64_t> &size,  // kernel_size 不设置默认值  
               const vector<int64_t> &stride,
               const vector<int64_t> &padding,
               /* dilation: _size_2_t = 1, */
               const vector<int64_t> &dilation,
               int8_t return_indices,
               int8_t ceil_mode,
               int8_t count_include_pad,
               int64_t divisor_override,
               const string padding_mode
               /* device */
               /* dtype */):
    _size(size),
    _stride(stride),
    _padding(padding),
    _dilation(dilation),
    _return_indices(return_indices),
    _ceil_mode(ceil_mode),
    _count_include_pad(count_include_pad),
    _divisor_override(divisor_override),
    _padding_mode(stringToPM(padding_mode))
{
    LogDebug("ConvNd(...)");
    _size.truncateArr(_dimsNum - 1);
    _stride.truncateArr(_dimsNum - 1);
    _padding.truncateArr(_dimsNum - 1);
    _dilation.truncateArr(_dimsNum - 1);
    // _idx = NETWORK->attachLayer(this);
    // EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

#if LIBTORCH_EXT
void PoolNd::paramLoad(const string &str, const at::Tensor &data) {
    LogDebug("PoolNd::Load_param(const std::string &, const at::Tensor &)");
    if (!strcmp(str.c_str(), "inputs")) {
        initialize(data);
    }
}

void PoolNd::initialize(const at::Tensor &data) {
    LogDebug("PoolNd::initialize(const at::Tensor &)");
    // EXIT_ERROR_CHECK(data.dtype() == at::kFloat, false, "type in data if not FLOATCA");
    int64_t dims_tmp[MAX_CONV_DIMENSIONS] = {0};
    get_tensor_dimensions(data, dims_tmp);
    _batch = getDimsIdx(dims_tmp, 1);
    _inputs = static_cast<int64_t>(data.numel()) / _batch;
    setDims(_dims, dims_tmp, SizeTToInt64(static_cast<size_t>(_dimsNum)));     // _dims
    setOutDims(_dims, _out_dims, _dims[1], 
               _size.getVector(), 
               _stride.getVector(),
               _padding.getVector());      // _out_dims
    Layer::setOutputs(_out_dims); 
    // POOL 不计算 _workspace_szie
    updateNet();
}
PoolNd::PoolNd(SerPoolPrePack &&spp):
    _size(std::move(std::get<1>(spp))),
    _stride(std::move(std::get<2>(spp))),
    _padding(std::move(std::get<3>(spp))),
    _dilation(std::move(std::get<4>(spp))),
    _return_indices(std::get<5>(spp)),
    _ceil_mode(std::get<6>(spp)),
    _count_include_pad(std::get<7>(spp)),
    _divisor_override(std::get<8>(spp)),
    _padding_mode(static_cast<PADDING_MODE>(int64ToSizeT(std::get<9>(spp))))
{
    LogDebug("PoolNd(SerPoolPrePack &&)");
    _size.truncateArr(_dimsNum - 1);
    _stride.truncateArr(_dimsNum - 1);
    _padding.truncateArr(_dimsNum - 1);
    _dilation.truncateArr(_dimsNum - 1);
}
const SerPoolPrePack &PoolNd::makePack() {
    LogDebug("makePack() for PoolNd");
    SerPoolPrePack *pspp = new SerPoolPrePack(
        std::move(makeLPack()),
        std::move(_size.getVector()),
        std::move(_stride.getVector()),
        std::move(_padding.getVector()),
        std::move(_dilation.getVector()),
        _return_indices,
        _ceil_mode,
        _count_include_pad,
        _divisor_override,
        static_cast<int64_t>(_padding_mode));
    return *pspp;
}
#endif // LIBTORCH_EXT
#if TORCHZONE
void PoolNd::make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION) const {
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

    make_pool_ca(invitation,
                 static_cast<uint32_t>(_type),
                 _idx,
                 size,
                 stride,
                 padding,
                 dilation,
                 _return_indices,
                 _ceil_mode,
                 _count_include_pad,
                 Int64TToINTTA(_divisor_override),
                 static_cast<uint32_t>(_padding_mode));
}
#endif 

void PoolNd::print() const {
    cout << "!!!!!!!!!!!!!! print() :  PoolNd !!!!!!!!!!!!!!" << endl \
         << "idx            : " << (getIdx()) << endl \
         << "type           : " << _type << endl \
         << "keepIn         : " << ((_keepIn) ? "ture" : "false") << endl \
         << "keepOut        : " << ((_keepOut) ? "ture" : "false") << endl \
         << "batch          : " << _batch << endl \
         << "inputs         : " << _inputs << endl \
         << "outputs        : " << _outputs << endl;

    cout << "dims           : ";
    for (int64_t i = 1; i <= _dims[0]; ++i) cout << _dims[i] << " ";
    cout << endl;

    cout << "out_dims       : ";
    for (int64_t i = 1; i <= _out_dims[0]; ++i) cout << _out_dims[i] << " ";
    cout << endl;

    cout << "size           : ";
    _size.print();
    cout << "stride         : ";
    _stride.print();
    cout << "padding        : ";
    _padding.print();
    cout << "dilation       : ";
    _dilation.print();

    cout << "return_indices : " << (_return_indices ? "true" : "false") << endl;
    cout << "ceil_mode      : " << (_ceil_mode ? "true" : "false") << endl;
    cout << "count_include_pad     : " << (_count_include_pad ? "true" : "false") << endl;
    cout << "divisor_override      : " <<  _divisor_override << endl;
    cout << "padding_mode   : " << _padding_mode << endl \
         << "dimsNum        : " << _dimsNum << endl;
}

// ==============================================================
MaxPool2d::MaxPool2d(const vector<int64_t> &size,  // kernel_size 不设置默认值  
          const vector<int64_t> &stride,
          const vector<int64_t> &padding,
          /* dilation: _size_2_t = 1, */
          const vector<int64_t> &dilation,
          int8_t return_indices,
          int8_t ceil_mode):
    Layer(MAXPOOL_TYPE, D2),
    PoolNd(size, stride, padding, dilation, return_indices, ceil_mode, false, INVALID_VALUE_U, "ZEROS")
{
    LogDebug("MaxPool2d(...)");
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

MaxPool2d::~MaxPool2d() {
    LogDebug("~MaxPool2d()");
}
#if LIBTORCH_EXT
MaxPool2d::MaxPool2d(SerPoolPrePack spp):
    Layer(std::move(std::get<0>(spp))),
    PoolNd(std::move(spp))
{
    LogDebug("MaxPool2d(SerPoolPrePack)");
    int32_t ret = NETWORK->attachLayer(this, _idx);
    // cout << "ret == " << ret << endl;
    // EXIT_ERROR_CHECK(ret == _idx, false, "error idx from OpsCtx");
    updateNet();
}

void MaxPool2d::operator()(at::Tensor &input) {
    LogDebug("void MaxPool2d::operator()(at::Tensor &)");
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
}
void MaxPool2d::operator()() {
    LogDebug("void MaxPool2d::operator()()");
    NETWORK->registerIdx(_idx);
}

void MaxPool2d::attach(at::Tensor &input) {
    LogDebug("void MaxPool2d::attach(at::Tensor &) : idx = %d : ", _idx);
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
    // return this;
}
void MaxPool2d::attach() {
    LogDebug("void MaxPool2d::attach() : idx = %d : ", _idx);
    NETWORK->registerIdx(_idx);
    // return this;
}
#endif // LIBTORCH_EXT

// ==============================================================
AvgPool2d::AvgPool2d(const vector<int64_t> &size,  // kernel_size 不设置默认值  
          const vector<int64_t> &stride,
          const vector<int64_t> &padding,
          int8_t ceil_mode,
          int8_t count_include_pad,
          int64_t divisor_override):
    Layer(AVGPOOL_TYPE, D2),
    PoolNd(size, stride, padding, vector<int64_t>(), false, ceil_mode, count_include_pad, divisor_override, "ZEROS")
{
    LogDebug("AvgPool2d(...)");
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

AvgPool2d::~AvgPool2d() {
    LogDebug("~AvgPool2d()");
}
#if LIBTORCH_EXT
AvgPool2d::AvgPool2d(SerPoolPrePack spp):
    Layer(std::move(std::get<0>(spp))),
    PoolNd(std::move(spp))
{
    LogDebug("AvgPool2d(SerPoolPrePack)");
    int32_t ret = NETWORK->attachLayer(this, _idx);
    // cout << "ret == " << ret << endl;
    // EXIT_ERROR_CHECK(ret == _idx, false, "error idx from OpsCtx");
    updateNet();
}

void AvgPool2d::operator()(at::Tensor &input) {
    LogDebug("void AvgPool2d::operator()(at::Tensor &)");
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
}
void AvgPool2d::operator()() {
    LogDebug("void AvgPool2d::operator()()");
    NETWORK->registerIdx(_idx);
}

void AvgPool2d::attach(at::Tensor &input) {
    LogDebug("void AvgPool2d::attach(at::Tensor &) : idx = %d : ", _idx);
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
    // return this;
}
void AvgPool2d::attach() {
    LogDebug("void AvgPool2d::attach() : idx = %d : ", _idx);
    NETWORK->registerIdx(_idx);
    // return this;
}
#endif // LIBTORCH_EXT

// ==============================================================
} // namespace end of core
} // namespace end of PyTZone