#include "Linear.h"
#include <RunMgr.h>

namespace PyTZone {
namespace core {

Linear::~Linear() {
    LogDebug("~Linear()");
    destory();
}
Linear::Linear(int64_t in_c, int64_t out_c, int8_t isBias):
    Layer(FCONNECTED_TYPE, DN),
    _in_c(in_c),
    _out_c(out_c),
    _isBias(isBias)
{
    LogDebug("Linear(...)");
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");

    _nweights = _in_c * _out_c;
    if (_isBias) _nbiases = _out_c;
}

void Linear::print() const {
    cout << "!!!!!!!!!!!!!! print() :  Linear !!!!!!!!!!!!!!" << endl \
         << "idx            : " << (getIdx()) << endl \
         << "type           : " << _type << endl \
         << "keepIn         : " << ((_keepIn) ? "ture" : "false") << endl \
         << "keepOut        : " << ((_keepOut) ? "ture" : "false") << endl \
         << "batch          : " << _batch << endl \
         << "inputs         : " << _inputs << endl \
         << "outputs        : " << _outputs << endl \
         << "nweights       : " << _nweights << endl \
         << "nbiases        : " << _nbiases << endl;
         
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

    cout << "in_c           : " << _in_c << endl \
         << "out_c          : " << _out_c << endl;
    cout << "isBias         : " << (_isBias ? "true" : "false") << endl;

    return;
}

void Linear::memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const {
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

#if LIBTORCH_EXT
void Linear::paramLoad(const string &str, const at::Tensor &data) {
    LogDebug("Linear::Load_param(const std::string &, const at::Tensor &)");
    paramLoad_for_datas(str, data);
    if (!strcmp(str.c_str(), "inputs")) {
        initialize(data);
    }
}

void Linear::initialize(const at::Tensor &data) {
    LogDebug("Linear::initialize(const at::Tensor &)");
    // EXIT_ERROR_CHECK(data.dtype() == at::kFloat, false, "type in data if not FLOATCA");
    int64_t dims_tmp[MAX_CONV_DIMENSIONS] = {0};
    get_tensor_dimensions(data, dims_tmp);
    _batch = getDimsIdx(dims_tmp, 1);
    EXIT_ERROR_CHECK(_batch * _in_c == data.numel(), false, "input tensor dimension error");
    _inputs = _in_c;
    _outputs = _out_c;
    
    setDims(_dims, dims_tmp, dims_tmp[0] - 1);     // _dims
    setDims(_out_dims, dims_tmp, dims_tmp[0] - 1);
    _out_dims[1] = _out_c;

    updateNet();
}
Linear::Linear(SerLinearPrePack spp):
    Layer(std::move(std::get<0>(spp))),
    _in_c(std::get<1>(spp)),
    _out_c(std::get<2>(spp)),
    _isBias(std::get<3>(spp))
{
    LogDebug("Linear(SerLinearPrePack)");
    int32_t ret = NETWORK->attachLayer(this, _idx);
    // cout << "ret == " << ret << endl;
    // EXIT_ERROR_CHECK(ret == _idx, false, "error idx from OpsCtx");
    updateNet();
}
const SerLinearPrePack &Linear::makePack() {
    LogDebug("makePack() for Linear");
    SerLinearPrePack *pspp = new SerLinearPrePack(
        std::move(makeLPack()),
        _in_c, _out_c, _isBias);
    return *pspp;
}
void Linear::operator()(at::Tensor &input) {
    LogDebug("void Linear::operator()(at::Tensor &)");
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
}
void Linear::operator()() {
    LogDebug("void Linear::operator()()");
    NETWORK->registerIdx(_idx);
}

void Linear::attach(at::Tensor &input) {
    LogDebug("void Linear::attach(at::Tensor &) : idx = %d : ", _idx);
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
    // return this;
}
void Linear::attach() {
    LogDebug("void Linear::attach() : idx = %d : ", _idx);
    NETWORK->registerIdx(_idx);
    // return this;
}

#endif // LIBTORCH_EXT

#if TORCHZONE
void Linear::make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION) const {
    LogDebug("make_layer_ext_CA(TEEC_INVITATION_T * = nullptr) : idx = %d", _idx);
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);
    make_linear_ca(invitation,
                   static_cast<uint32_t>(_type),
                   _idx,
                   Int64TToINTTA(_in_c),
                   Int64TToINTTA(_out_c),
                   _isBias);
}
#endif

} // namespace end of core
} // namespace end of PyTZone