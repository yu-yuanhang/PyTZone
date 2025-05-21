#include "NormBatch.h"
#include <RunMgr.h>

namespace PyTZone {
namespace core {

NormNd::~NormNd() {
    LogDebug("~NormNd()");
    destory();
}

NormNd::NormNd(int64_t in_c,   // 取决于 Norm 的类型
               double eps,
               double momentum,
               int8_t affine,
               int8_t track_running_stats
               /* device */
               /* dtype */):
    _in_c(in_c),
    _eps(eps),
    _momentum(momentum),
    _affine(affine),
    _track_running_stats(track_running_stats)
{
    LogDebug("NormNd(...)");
    // _idx = NETWORK->attachLayer(this);
    // EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
    _nweights = _nbiases = _in_c;
}

#if LIBTORCH_EXT
void NormNd::paramLoad(const string &str, const at::Tensor &data) {
    LogDebug("NormNd::Load_param(const std::string &, const at::Tensor &)");
    paramLoad_for_datas(str, data);
    if (!strcmp(str.c_str(), "inputs")) {
        initialize(data);
    }
}
void NormNd::initialize(const at::Tensor &data) {
    LogDebug("NormNd::initialize(const at::Tensor &)");
    // EXIT_ERROR_CHECK(data.dtype() == at::kFloat, false, "type in data if not FLOATCA");
    // setBatch(data);
    int64_t dims_tmp[MAX_CONV_DIMENSIONS] = {0};
    get_tensor_dimensions(data, dims_tmp);
    _batch = getDimsIdx(dims_tmp, 1);
    _outputs = _inputs = static_cast<int64_t>(data.numel()) / _batch;
    setDims(_dims, dims_tmp, SizeTToInt64(static_cast<size_t>(_dimsNum)));     // _dims
    setDims(_out_dims, dims_tmp, SizeTToInt64(static_cast<size_t>(_dimsNum)));
    updateNet();
}

NormNd::NormNd(SerNormNdPrePack &&spp):
    _in_c(std::get<1>(spp)),
    _eps(std::get<2>(spp)),
    _momentum(std::get<3>(spp)),
    _affine(std::get<4>(spp)),
    _track_running_stats(std::get<5>(spp))
{
    LogDebug("NormNd(SerNormNdPrePack &&)");
}

const SerNormNdPrePack &NormNd::makePack() {
    LogDebug("makePack() for NormNd");
    SerNormNdPrePack *pspp = new SerNormNdPrePack(
        std::move(makeLPack()),
        _in_c, _eps, _momentum, _affine, _track_running_stats);
    return *pspp;
}

#endif // LIBTORCH_EXT

#if TORCHZONE
void NormNd::make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION) const {
    LogDebug("make_layer_ext_CA(TEEC_INVITATION_T * = nullptr) : idx = %d", _idx);
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);
    make_norm_ca(invitation,
                 static_cast<uint32_t>(_type),
                 _idx,
                 Int64TToINTTA(_in_c),
                 _eps,
                 _momentum,
                 _affine,
                 _track_running_stats,
                 _mean, _variance,
                 _nweights, _nweights);
}
#endif 

void NormNd::print() const {
    cout << "!!!!!!!!!!!!!! print() :  NormNd !!!!!!!!!!!!!!" << endl \
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
    if (nullptr != _biases) {
        for (int64_t i = 0; i < _nbiases && i < 40; ++i) {
            if (!(i % 10) && 0 != i) cout << endl;
            printf("%+.4f  ", _biases[i]);
        }
    }
    cout << endl;
    cout << "==========================================" << endl;
    cout << "means ..." << endl;
    if (nullptr != _mean) {
        for (int64_t i = 0; i < _nweights && i < 40; ++i) {
            if (!(i % 10) && 0 != i) cout << endl;
            printf("%+.4f  ", _mean[i]);
        }
    }
    cout << endl;
    cout << "==========================================" << endl;
    cout << "variances ..." << endl;
    if (nullptr != _variance) {
        for (int64_t i = 0; i < _nweights && i < 40; ++i) {
            if (!(i % 10) && 0 != i) cout << endl;
            printf("%+.4f  ", _variance[i]);
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
         << "eps            : " << _eps << endl \
         << "momentum       : " << _momentum << endl;
    cout << "affine                 : " << (_affine ? "true" : "false") << endl;
    cout << "track_running_stats    : " << (_track_running_stats ? "true" : "false") << endl;
}

void NormNd::memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const {
    heapAll += _outputs;
    if (_keepIn) heapApply += _inputs;
    if (_keepOut) heapApply += _outputs;

    heapAll += _nweights;
    heapAll += _nbiases;
    heapAll += _nweights;
    heapAll += _nweights;

    heapApply += _nweights;
    heapApply += _nbiases;
    heapApply += _nweights;
    heapApply += _nweights;

    heapWeightsOnly += _nweights;
    heapWeightsOnly += _nbiases;
    heapWeightsOnly += _nweights;
    heapWeightsOnly += _nweights;
}

// ==============================================================
BatchNorm2d::BatchNorm2d(int64_t in_c,
                         double eps,
                         double momentum,
                         int8_t affine,
                         int8_t track_running_stats):
    Layer(BATCHNORM_TYPE, D2),
    NormNd(in_c, eps, momentum, affine, track_running_stats)
{
    LogDebug("BatchNorm2d(...)");
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
}

BatchNorm2d::~BatchNorm2d() {
    LogDebug("~BatchNorm2d()");
}

#if LIBTORCH_EXT
BatchNorm2d::BatchNorm2d(SerNormNdPrePack spp):
    Layer(std::move(std::get<0>(spp))),
    NormNd(std::move(spp))
{
    LogDebug("BatchNorm2d(SerNormNdPrePack)");
    int32_t ret = NETWORK->attachLayer(this, _idx);
    // cout << "ret == " << ret << endl;
    // EXIT_ERROR_CHECK(ret == _idx, false, "error idx from OpsCtx");
    updateNet();
}

void BatchNorm2d::operator()(at::Tensor &input) {
    LogDebug("void BatchNorm2d::operator()(at::Tensor &)");
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
}
void BatchNorm2d::operator()() {
    LogDebug("void BatchNorm2d::operator()()");
    NETWORK->registerIdx(_idx);
}

void BatchNorm2d::attach(at::Tensor &input) {
    LogDebug("void BatchNorm2d::attach(at::Tensor &) : idx = %d : ", _idx);
    // 将要调入 TA 的第一层 重置 net 的输入 和 idx 列表
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
    // return this;
}
void BatchNorm2d::attach() {
    LogDebug("void BatchNorm2d::attach() : idx = %d : ", _idx);
    NETWORK->registerIdx(_idx);
    // return this;
}
#endif // LIBTORCH_EXT

} // namespace end of core
} // namespace end of PyTZone