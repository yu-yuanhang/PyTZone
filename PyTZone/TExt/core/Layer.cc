#include "Layer.h"
#include "./utils/Serializer.h"
#include <network.h>
#include <RunMgr.h>

namespace PyTZone {
namespace core {
// ===================================================

PADDING_MODE stringToPM(const string &str) {
    if (str == "ZEROS" || str == "zeros") return PADDING_MODE::ZEROS_PADDING;
    // else if 
    // else if 
    else {
        LogWarn("error PADDING_MODE : str = %s", str.c_str());
        return PADDING_MODE::ZEROS_PADDING; 
    }
}
bool isParams(LAYER_TYPE type) {
    return (type < ACTIV_TYPE ? true : false);
}
// =========================================================
void Layer::setData_randn_t() {
    // 随机数生成器
    // std::default_random_engine generator(static_cast<unsigned int>(std::time(nullptr)));
    std::default_random_engine generator(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<FLOATCA> distribution(-1.0f, 1.0f);  // 生成范围在 -1.0 到 1.0 之间的随机数

    if (nullptr != _weights) free(_weights);
    _weights == nullptr;
    _weights = (FLOATCA *)malloc(_nweights * FLOAT_SIZE);
    if (nullptr == _weights) {
        std::cout << "malloc for weights error : nweights = " << _nweights << std::endl;
        return;
    }
    for (int i = 0; i < _nweights; ++i) {_weights[i] = distribution(generator);}

    if (nullptr != _biases) free(_biases);
    _biases == nullptr;
    _biases = (FLOATCA *)malloc(_nbiases * FLOAT_SIZE);
    if (nullptr == _biases) {
        std::cout << "malloc for biases error : nbiases = " << _nbiases << std::endl;
        return;
    }
    for (int i = 0; i < _nbiases; ++i) {_biases[i] = distribution(generator);}

    if (BATCHNORM_TYPE == _type) {
        if (nullptr != _mean) free(_mean);
        _mean == nullptr;
        _mean = (FLOATCA *)malloc(_nweights * FLOAT_SIZE);
        if (nullptr == _mean) {
            std::cout << "malloc for mean error : nweights = " << _nweights << std::endl;
            return;
        }
        for (int i = 0; i < _nweights; ++i) {_mean[i] = distribution(generator);}

        if (nullptr != _variance) free(_variance);
        _variance == nullptr;
        _variance = (FLOATCA *)malloc(_nweights * FLOAT_SIZE);
        if (nullptr == _variance) {
            std::cout << "malloc for variance error : nweights = " << _nweights << std::endl;
            return;
        }
        for (int i = 0; i < _nweights; ++i) {_variance[i] = 1;}
    }
}

// int32_t idx_Layer_Global = 0;
Layer::Layer(LAYER_TYPE type, DIMENSIONALITY dimsNum):
    _activate(nullptr),
    _type(type),
    _dimsNum(dimsNum),
    _dims{},
    _out_dims{},
    _binary(INVALID_VALUE_U),
    _xnor(INVALID_VALUE_U),
    _keepIn(INVALID_VALUE_U),
    _keepOut(INVALID_VALUE_U),
    _mean(nullptr),
    _variance(nullptr),
    _batch(1),  // 通常初始化为 1 或者 32
    _inputs(INVALID_VALUE_U),
    _outputs(INVALID_VALUE_U),
    _weights(nullptr),
    _biases(nullptr),
    // _weights(at::empty({})),
    // _biases(at::empty({})),
    _nweights(INVALID_VALUE_U),
    _nbiases(INVALID_VALUE_U),
    // _output(nullptr),
    // _delta(nullptr), 
    /* _batch_normalize(INVALID_VALUE_U), */
    /* _learning_rate_scale(1.0),  // 默认不需要缩放 */
    /* _cost(nullptr), */
    /* _random(INVALID_VALUE_U), */
    /* _isloss(INVALID_VALUE_U), */
    _workspace_size(INVALID_VALUE_U),
    _idx(INT32_MAX),
    _layer1(nullptr),_layer2(nullptr),
    _input(nullptr),_output(nullptr)
{
    LogDebug("Layer(LAYER_TYPE)");
    // _layer1 = nullptr;
    // _layer2 = nullptr;
}
#if SERIALIZER
// ...... todo
Layer::Layer(ConvOpsCtx &coc):
    _type(static_cast<LAYER_TYPE>(int64ToSizeT(getForLayer(coc.get_params(), TYPE)))),
    _batch(getForLayer(coc.get_params(), BATCH)),  // 通常初始化为 1 或者 32
    _inputs(getForLayer(coc.get_params(), INPUTS)),
    _outputs(getForLayer(coc.get_params(), OUTPUTS)),
    // _weights(nullptr),
    // _biases(nullptr),
    _weights(std::move(coc.get_weights())),
    _biases(coc.isBias() ? std::move((coc.get_bias()).value()) :  at::empty({})),
    _nweights(getForLayer(coc.get_params(), NWEIGHTS)),
    _nbiases(getForLayer(coc.get_params(), NBIASES)),
    // _output(nullptr),
    // _delta(nullptr), 
    /* _mean(nullptr), */
    /* _variance(nullptr), */
    /* _batch_normalize(INVALID_VALUE_U), */
    /* _learning_rate_scale(1.0),  // 默认不需要缩放 */
    /* _cost(nullptr), */
    /* _random(INVALID_VALUE_U), */
    /* _isloss(INVALID_VALUE_U), */
    _workspace_size(getForLayer(coc.get_params(), WORKSPACE_SIZE)),
    _idx(getForLayer(coc.get_params(), IDX))
{
    LogDebug("Layer(ConvOpsCtx &)");
}
#endif // SERIALIZER
#if LIBTORCH_EXT
Layer::Layer(SerPrePack_Layer &&sppL):
    _activate(nullptr),
    _type(static_cast<LAYER_TYPE>(int64ToSizeT(std::get<0>(sppL).at(TYPE)))),
    _dimsNum(static_cast<DIMENSIONALITY>(int64ToSizeT(std::get<0>(sppL).at(DIMSNUM)))),
    _dims{},
    _out_dims{},
    _binary(std::get<0>(sppL).at(BINARY) ? true : false),
    _xnor(std::get<0>(sppL).at(XNOR) ? true : false),
    _keepIn(std::get<0>(sppL).at(KEEPIN) ? true : false),
    _keepOut(std::get<0>(sppL).at(KEEPOUT) ? true : false),
    _mean(nullptr),
    _variance(nullptr),
    _batch(std::get<0>(sppL).at(BATCH)),  // 通常初始化为 1 或者 32
    _inputs(std::get<0>(sppL).at(INPUTS)),
    _outputs(std::get<0>(sppL).at(OUTPUTS)),
    _weights(nullptr),
    _biases(nullptr),
    // _weights(std::move(std::get<1>(sppL))),
    // _biases(std::move(std::get<2>(sppL).value())),
    _nweights(std::get<0>(sppL).at(NWEIGHTS)),
    _nbiases(std::get<0>(sppL).at(NBIASES)),
    // _output(nullptr),
    // _delta(nullptr), 
    /* _batch_normalize(INVALID_VALUE_U), */
    /* _learning_rate_scale(1.0),  // 默认不需要缩放 */
    /* _cost(nullptr), */
    /* _random(INVALID_VALUE_U), */
    /* _isloss(INVALID_VALUE_U), */
    _workspace_size(std::get<0>(sppL).at(WORKSPACE_SIZE)),
    _idx(Int64TToInt32T(std::get<0>(sppL).at(IDX))),
    _layer1(nullptr),_layer2(nullptr),
    _input(nullptr),_output(nullptr)
{
    LogDebug("Layer(SerPrePack_Layer &&)");
    
    int64_t inNum = std::get<1>(sppL).size();
    int64_t outNum = std::get<2>(sppL).size();
    setDimsV(std::move(std::get<1>(sppL)), _dims, inNum);
    setDimsV(std::move(std::get<2>(sppL)), _out_dims, outNum);
    // 简单的值拷贝
    if (_nweights) {
        _weights = new FLOATCA[_nweights];
        memcpy(_weights, std::move(std::get<3>(sppL)).data_ptr<FLOATCA>(), _nweights * FLOAT_SIZE);
    }
    if (_nbiases) {
        _biases = new FLOATCA[_nbiases];
        memcpy(_biases, (std::move(std::get<4>(sppL).value())).data_ptr<FLOATCA>(), _nbiases * FLOAT_SIZE);
    }
    if (BATCHNORM_TYPE == _type) {
        _mean = new FLOATCA[_nweights];
        memcpy(_mean, std::move(std::get<5>(sppL)).data_ptr<FLOATCA>(), _nweights * FLOAT_SIZE);
        _variance = new FLOATCA[_nweights];
        memcpy(_variance, std::move(std::get<6>(sppL)).data_ptr<FLOATCA>(), _nweights * FLOAT_SIZE);
    }
    if (_keepIn) {_input = new FLOATCA[_inputs];}
    if (_keepOut) {_output = new FLOATCA[_outputs];}
}
const SerPrePack_Layer &Layer::makeLPack() const {
    // 确保内容的生命周期更长或是 直接返回的临时变量
    // std::make_tuple(std::move(makeParams()), std::move(_weights), std::move(std::optional<at::Tensor>(std::move(_biases)))),
    SerPrePack_Layer *spp_l = new SerPrePack_Layer(
                        std::move(makeParams()), 
                        std::move(arrayToVector<int64_t>((_dims + 1), _dims[0])),
                        std::move(arrayToVector<int64_t>((_out_dims + 1), _out_dims[0])),
                        at::from_blob(_weights, {_nweights}, at::kFloat), 
                        std::move(std::optional<at::Tensor>(at::from_blob(_biases, {_nbiases}, at::kFloat))),
                        at::from_blob(_mean, {_nweights}, at::kFloat), 
                        at::from_blob(_variance, {_nweights}, at::kFloat)); 
    return *spp_l;
}

void Layer::get_tensor_dimensions(const at::Tensor &data, int64_t dims[MAX_CONV_DIMENSIONS]) {
    auto sizes = data.sizes();
    EXIT_ERROR_CHECK(((sizes.size() + 1) > MAX_CONV_DIMENSIONS), true, "dimensions for inputs error");
    dims[0] = sizes.size();
    for (int64_t i = 1; i <= dims[0]; ++i)  dims[i] = sizes[i - 1];
    // 安全起见 处理剩余的维度
    for (int64_t i = dims[0] + 1; i < MAX_CONV_DIMENSIONS; ++i) dims[i] = 0; 
}
void Layer::setWeights(const at::Tensor &data) {
    LogDebug("Layer::setWeights(const at::Tensor &)");
    // EXIT_ERROR_CHECK(data.dtype() == at::kFloat, false, "type in data if not FLOATCA");
    // data.dtype() == at::kInt
    // 目前 暂不考虑 int 类型的情况
    if (!_nweights) return;
    int64_t nweights = static_cast<int64_t>(data.numel());
    EXIT_ERROR_CHECK((nweights != _nweights), true, "nweights set error");

    // EXIT_ERROR_CHECK(_nweights, SIZE_MAX, "nweights error");
    // _weights = data.data_ptr<FLOATCA>();
    if (nullptr != _weights) delete[] _weights;
    _weights = new FLOATCA[_nweights];
    memcpy(_weights, data.data_ptr<FLOATCA>(), _nweights * FLOAT_SIZE);
    // _weights = data;
}
void Layer::setBiases(const at::Tensor &data) {
    LogDebug("Layer::setBiases(const at::Tensor &)");
    // EXIT_ERROR_CHECK(data.dtype() == at::kFloat, false, "type in data if not FLOATCA");
    // EXIT_ERROR_CHECK(_isBias, false, "bias == flase");
    if (!_nbiases) return;
    int64_t nbiases = static_cast<int64_t>(data.numel());
    EXIT_ERROR_CHECK((nbiases != _nbiases), true, "nbiases set error");

    // EXIT_ERROR_CHECK(_nbiases, SIZE_MAX, "nbiases error");
    // _biases = data.data_ptr<FLOATCA>();
    if (nullptr != _biases) delete[] _biases;
    _biases = new FLOATCA[_nbiases];
    memcpy(_biases, data.data_ptr<FLOATCA>(), _nbiases * FLOAT_SIZE);
    // _biases = data;
    // _isBias = true;
}
void Layer::setMeans(const at::Tensor &data) {
    LogDebug("Layer::setMeans(const at::Tensor &)");
    if(!_nweights) _nweights = static_cast<int64_t>(data.numel());
    if (nullptr != _mean) delete[] _weights;
    _mean = new FLOATCA[_nweights];
    memcpy(_mean, data.data_ptr<FLOATCA>(), _nweights * FLOAT_SIZE);
}
void Layer::setVariances(const at::Tensor &data) {
    LogDebug("Layer::setVariances(const at::Tensor &)");
    if(!_nweights) _nweights = static_cast<int64_t>(data.numel());
    if (nullptr != _variance) delete[] _weights;
    _variance = new FLOATCA[_nweights];
    memcpy(_variance, data.data_ptr<FLOATCA>(), _nweights * FLOAT_SIZE);
}
void Layer::setBatch(const at::Tensor &data) {
    int64_t dims_tmp[MAX_CONV_DIMENSIONS] = {0};
    get_tensor_dimensions(data, dims_tmp);
    _batch = getDimsIdx(dims_tmp, 1);
}
void Layer::paramLoad_for_datas(const std::string &str, const at::Tensor &data) {
    LogDebug("Layer::paramLoad_for_datas(const std::string &, const at::Tensor &)");
    // 基本的检查 : name (其实这个逻辑基本是交给 上层封装)
    // 这里仅仅需要判断 weight 和 bias
    // 关于算子定位的逻辑交给 network 的封装
    
    // auto [first, second] = splitString_dot(str.c_str());
    // return std::make_tuple(str, "NULL")
    // EXIT_ERROR_CHECK(second, "NULL", "Conv2d::Load_param(...) : name error");
    // EXIT_ERROR_CHECK(checkStr(first), false, "Conv2d::Load_param(...) : name error");

    // 主要关于 python 脚本文件中的变量的生命周期我不太懂
    // 如果为了安全起见 还是考虑 深拷贝
    if (!strcmp(str.c_str(), "weight")) {setWeights(data);}
    else if (!strcmp(str.c_str(), "bias")) {setBiases(data);} 
}
#endif // LIBTORCH_EXT
Layer::~Layer() {
    // printf("Layer::~Layer()\n");
    LogDebug("~Layer()");
    destory();
}
void Layer::destory() {
    if (nullptr != _weights) {delete _weights;_weights = nullptr;}
    if (nullptr != _biases) {delete _biases;_biases = nullptr;}
    if (nullptr != _mean) {delete _mean;_mean = nullptr;}
    if (nullptr != _variance) {delete _variance;_variance = nullptr;}
    if (nullptr != _input) {delete _input;_input = nullptr;}
    if (nullptr != _output) {delete _output;_output = nullptr;}
    _layer1 = nullptr;
    _layer2 = nullptr;
    // if (nullptr != _output) free(_output);
    // if (nullptr != _delta) free(_delta);
}

std::vector<int64_t> &Layer::makeParams() const {
    std::vector<int64_t> *pvec = new std::vector<int64_t>{
        static_cast<int64_t>(_type),
        static_cast<int64_t>(_dimsNum),
        static_cast<int64_t>(_binary),
        static_cast<int64_t>(_xnor),
        static_cast<int64_t>(_keepIn),
        static_cast<int64_t>(_keepOut),
        _batch,
        _inputs,
        _outputs,
        _nweights,
        _nbiases,
        _workspace_size,
        static_cast<int64_t>(_idx)};
    // 这里暂时并没有做过多的检查
    return *pvec;
}
#if SERIALIZER
const at::Tensor &Layer::getWeights() const {return _weights;}
const at::Tensor &Layer::getBiases() const {return _biases;}
#endif  // SERIALIZER
#if TORCHZONE
void Layer::make_layer_CA(TEEC_INVITATION_T *TEEC_INVITATION) const {
    LogDebug("make_layer_CA(TEEC_INVITATION_T * = nullptr) : idx = %d", _idx);

    INT_TA dims[MAX_CONV_DIMENSIONS];
    INT_TA out_dims[MAX_CONV_DIMENSIONS];
    for (size_t i = 0; i < MAX_CONV_DIMENSIONS; ++i) {
        dims[i] = Int64TToINTTA(_dims[i]);
        out_dims[i] = Int64TToINTTA(_out_dims[i]);  
    }

    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);
    make_layer_ca(invitation,
                  static_cast<uint32_t>(_type),
                  static_cast<uint32_t>(_dimsNum),
                  dims, out_dims,
                  _binary, _xnor, _keepIn, _keepOut,
                  Int64TToINTTA(_batch),
                  Int64TToINTTA(_inputs),
                  Int64TToINTTA(_outputs),
                  Int64TToINTTA(_nweights),
                  Int64TToINTTA(_nbiases),
                  Int64TToINTTA(_workspace_size),
                  _idx,
                  _weights, _biases);
}
#endif

void Layer::setDims(int64_t target[MAX_CONV_DIMENSIONS], 
                    int64_t source[MAX_CONV_DIMENSIONS],
                    int64_t num) {
    EXIT_ERROR_CHECK(num == source[0] - 1, false, "dimensions for inputs error");
    target[0] = num;
    for (int64_t i = 1; i <= num; ++i) target[i] = source[i + 1];
    for (int64_t i = num + 1; i < MAX_CONV_DIMENSIONS; ++i) target[i] = 0; 
}
void Layer::setDimsV(const std::vector<int64_t> &dims, 
                     int64_t target_dims[MAX_CONV_DIMENSIONS],
                     int64_t num) 
{
    if (dims.size()) {
        EXIT_ERROR_CHECK(num == dims.size(), false, "dimensions for inputs error");
        target_dims[0] = num;
        for (int64_t i = 1; i <= num; ++i) target_dims[i] = dims.at(i - 1);
        for (int64_t i = num + 1; i < MAX_CONV_DIMENSIONS; ++i) target_dims[i] = 0; 
    }
}

void Layer::setOutDims(int64_t dims[MAX_CONV_DIMENSIONS],
                int64_t out_dims[MAX_CONV_DIMENSIONS],
                int64_t num,
                vector<int64_t> size,
                vector<int64_t> stride,
                vector<int64_t> padding
                ) {
    out_dims[0] = dims[0];
    out_dims[1] = num;
    // set_conv_out_height();
    // set_conv_out_width();
    for (int64_t i = 2; i <= out_dims[0]; ++i) 
        out_dims[i] = ((dims[i] + 2 * padding.at(i - 2) - size.at(i - 2)) / stride.at(i - 2)) + 1;
}
void Layer::setOutputs(int64_t out_dims[MAX_CONV_DIMENSIONS]) {
    _outputs = 1;
    for (int64_t i = 1; i <= out_dims[0]; ++i) _outputs *= out_dims[i];
    // 安全考虑 检查是否越界
    EXIT_ERROR_CHECK(_outputs, INVALID_VALUE_U, "_outputs error");
}
void Layer::setWorkspaceSize(int64_t dims[MAX_CONV_DIMENSIONS],
                             int64_t out_dims[MAX_CONV_DIMENSIONS],
                             int64_t groups,
                             vector<int64_t> size,
                             DIMENSIONALITY dimsNum) {
    // return (int64_t)l.h*l.w*l.size*l.size*l.n*FLOAT_SIZE;
    // 这里的工作空间的算法我也有点参透不了含义
    // 需要对于卷积的底层计算过程进行分析才行

    _workspace_size = 1;
    for (int64_t i = 2; i < out_dims[0] + 1; ++i) 
        _workspace_size *= out_dims[i];
    
    for (size_t i = 0; i < (dimsNum - 1); ++i) 
        _workspace_size *= size.at(i);
    // 这里对于权重数据的类型目前还是只考虑 float32 的请情况
    _workspace_size *= (dims[1] * FLOAT_SIZE) / groups;
}

void Layer::registerPthr() const {NETWORK->setLayerThr(_type, _idx);}

void Layer::dataKeep(int64_t inputs, int64_t outputs, int64_t batch) {
    
    network *net = NETWORK;
    Layer *l = net->getLayer(0);
    
    if (_keepIn) {
        int64_t inputs_size = inputs * batch;
        if (nullptr != _input && 0 != inputs_size) 
            memcpy(_input, NETWORK->getInput(), inputs_size * FLOAT_SIZE);
    }
    if (_keepOut) {
        int64_t outputs_size = outputs * batch;
        if (nullptr != _output && 0 != outputs_size) 
            memcpy(_output, NETWORK->getOutput(), outputs_size * FLOAT_SIZE);
    }
}

void Layer::memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const {
    heapAll += _outputs;
    if (_keepIn) heapApply += _inputs;
    if (_keepOut) heapApply += _outputs;
}


} // namespace end of core
} // namespace end of PyTZone 
