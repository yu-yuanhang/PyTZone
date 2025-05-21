#include "network.h"
#include <core/Layer.h>
#include <core/ops/Conv.h>
#include <core/ops/Activ.h>
#include <core/ops/Pool.h>
#include <core/ops/Linear.h>
#include <core/ops/NormBatch.h>
#include <core/ops/tsops.h>
#include <RunMgr.h>


namespace PyTZone {
namespace core {

int8_t Flag_TorchZone = (TORCHZONE ? true : false);

void Layer::updateNet() {
    NETWORK->updateWorkspaceSize(_workspace_size);
    NETWORK->updateInputs(_inputs);
    NETWORK->updateOutputs(_outputs);
}

network::network():
    _num(INVALID_VALUE_U),
    _subNum(INVALID_VALUE_U),
    _batch(1),  // 这里和 layer 同步
    _inputs(INVALID_VALUE_U),
    _outputs(INVALID_VALUE_U),
    _inoutSize(INVALID_VALUE_U),
    // _high(INVALID_VALUE_U),
    // _weight(INVALID_VALUE_U),
    // _channel(INVALID_VALUE_U),
    _workspace_size(INVALID_VALUE_U),
    _input(nullptr),
    _output(nullptr),
    _mem1(nullptr),
    _mem2(nullptr),
    _workspace(nullptr),
    _clip(INVALID_VALUE_U),
    _train(INVALID_VALUE_U),
    // _Layers 大小为 16 每个元素为 nullptr
    _Layers(INVALID_VALUE_U, nullptr),
    _index{},
    _layerThr{},
    _pre_outputs(INVALID_VALUE_U),
    _preLayer(nullptr),
    _tmp_outputs(INVALID_VALUE_U),
    _tmp_dims{}
{
    LogDebug("network(...)");
    resetIndex();
    std::fill(std::begin(_layerThr), std::end(_layerThr), INVALID_VALUE);
    // 注册 net 到 RunMgr
    // int32_t ret = RUNMGR->attachNet(this, MAIN_NET);
    // LogDebug("RUNMGR->attachNet(...) : ret = %d", ret);
}

network::~network() {
    // printf("network::~network()\n");
    if (nullptr != _output) {
        free(_output);
        _output = nullptr;
    }
    if (nullptr != _input) {
        free(_input);
        _input = nullptr;
    }
    if (nullptr != _workspace) {
        free(_workspace);
        _workspace = nullptr;
    }
    _mem1 = nullptr;
    _mem2 = nullptr;

    _preLayer = nullptr;
}

void network::print() const {
    cout << "!!!!!!!!!!!!!! print() :  netowrk_main !!!!!!!!!!!!!!" << endl \
         << "num                : " << _num << endl \
         << "subNum             : " << _subNum << endl \
         << "batch              : " << _batch << endl \
         << "inputs             : " << _inputs << endl \
         << "outputs            : " << _outputs << endl \
         << "inoutSize          : " << _inoutSize << endl \
         << "workspace_size     : " << _workspace_size << endl \
         << "clip               : " << _clip << endl \
         << "train              : " << (_train ? "tree" : "false") << endl;
    for (size_t i = 0; i < _Layers.size(); ++i) {
        Layer *player = _Layers[i];
        player->print();
    }
}
void network::printOutput(size_t noutput) const {
    cout << "==========================================" << endl;
    cout << "output ..." << endl;
    for (size_t i = 0; i < noutput && i < 40; ++i) {
        if (!(i % 10) && 0 != i) cout << endl;
        printf("%+.4f  ", _output[i]); // "+":始终显示符号
    }
    cout << endl;
    cout << "==========================================" << endl;
}

void network::memUsage_heap() const {
    int64_t heapAll = 0;
    int64_t heapApply = 0;
    int64_t heapWeightsOnly = 0;

    heapAll += _workspace_size;

    heapApply += _workspace_size;
    heapApply += _inoutSize;
    heapApply += _inoutSize;

    for (auto *l : _Layers) {l->memUsage_heap(heapAll, heapApply, heapWeightsOnly);}
    cout << "workspaceSize      = " << _workspace_size << endl;
    cout << "heapAll            = " << heapAll << endl;
    cout << "heapApply          = " << heapApply << endl;
    cout << "heapWeightsOnly    = " << heapWeightsOnly << endl;
}

void printNet() {
#if TORCHZONE 
    std::cout << "TORCHZONE = " << TORCHZONE << std::endl;
#else 
    std::cout << "TORCHZONE not define" << std::endl;
#endif
    NETWORK->print();
}

void network::setLayerThr(LAYER_TYPE type, int32_t idx) {
    switch (type)
    {
    case CONV_TYPE:         // 一般以卷积核为粒度来进行分割 
        break;
    case FCONNECTED_TYPE:   // 考虑以输出尺寸为粒度进行分割
        _layerThr[idx] = 2;
        break;
    case BATCHNORM_TYPE:
        break;
    case ACTIV_TYPE:
        break;
    case TSTATION_TYPE:     // 目前只考虑到 ADD 操作
        _layerThr[idx] = 1;
        break;
    default:
        _layerThr[idx] = 0;
        // LogError("The current type operator does not support multiple TAs.");
        break;
    }
}
size_t network::get_ConvMax_fromThr() const {
    int32_t idx = 0;
    int64_t maxNum = 0;
    std::for_each(_Layers.begin(), _Layers.end(), 
    [&maxNum, &idx, this](Layer *l) {
        if (0 != _layerThr[idx++] && CONV_TYPE == l->getType()) {
            int64_t numTmp = l->getNum_ConvNd();
            maxNum = (numTmp > maxNum ? numTmp : maxNum);
        }
    });
    return int64ToSizeT(maxNum);
}
size_t network::get_ConvMin_fromThr() const {
    int32_t idx = 0;
    int64_t minNum = INT64_MAX;
    std::for_each(_Layers.begin(), _Layers.end(), 
    [&minNum, &idx, this](Layer *l) {
        if (0 != _layerThr[idx++] && CONV_TYPE == l->getType()) {
            int64_t numTmp = l->getNum_ConvNd();
            minNum = (numTmp < minNum ? numTmp : minNum);
        }
    });
    return INT64_MAX == minNum ? 0 : int64ToSizeT(minNum);
}

void network::makeNet() {
    LogDebug("RUNMGR->makeNet() : Layers number = %d", _num);

    _input = (FLOATCA *)malloc(int64ToSizeT(_inoutSize * _batch) * FLOAT_SIZE);
    if (nullptr == _input) return;
    _mem1 = _input;
    _output = (FLOATCA *)malloc(int64ToSizeT(_inoutSize * _batch) * FLOAT_SIZE);
    if (nullptr == _output) return;
    _mem2 = _output;
    
    _workspace = (FLOATCA *)malloc(int64ToSizeT(_workspace_size));
    if (nullptr == _workspace) return;
}

void network::swapInOutPtr() {
    FLOATCA *tmp = _input;
    _input = _output;
    _output = tmp;
    // FLOATCA *tmp = NET_INPUT;
    // NET_INPUT = NET_OUTPUT;
    // NET_OUTPUT = tmp;    
}

// void network::resetInOutPtr() {
//     _input = _mem1 = NET_INPUT;
//     _output = _mem2 = NET_OUTPUT;
// }

void network::callForward() {

LogDebug("RUNMGR->Forward() : %d layers will be executed", _index[0]);
    for (int32_t i = 1; i <= _index[0]; ++i) {
        Layer* l = _Layers.at(_index[i]);
        LogDebug("layer.forward(...) : idx = %d", l->getIdx());
        l->forward();
        swapInOutPtr();
        if (TSTATION_TYPE <= l->getType()) continue;
        _preLayer = l;
        _pre_outputs = l->getOutputs();
    }
    swapInOutPtr();
}

#if TORCHZONE
int network::Forward_Fetch_CA(TEEC_INVITATION_T *TEEC_INVITATION) {
    LogDebug("RUNMGR->Forward_Fetch_CA() : %d layers will be executed", _index[0]);
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);

    // Layer *fl = getFirstL();
    int64_t inputs_L = _tmp_outputs;
    Layer *ll = getLastL();
    int64_t outputs_L = (nullptr == ll ? _tmp_outputs : ll->getOutputs());

#if 1   // 单线程单 TA 执行
    int8_t ret = forwardFetch_network_ca(invitation,
                                         _input, (inputs_L * _batch), 
                                         _output, (outputs_L * _batch), 
                                         _index);
#else   // 多线程多 TA 执行
    // 对于多线程的支持 各个线程的子任务划分目前安排在运行时
    // 这样的好处是减少了模型构建过程中需要由程序员指定的初始化代码

    // 为了避免执行过程中频繁的线程创建和销毁的开销过大
    // 这里仍然采用一个简单的 threadpool + taskqueue 的模式


    // 子任务划分需要遍历 _index 以获取层与层之间输入输出的依赖关系


#endif 

    return 0;
}

int network::callForward_CA(TEEC_INVITATION_T *TEEC_INVITATION) {
    LogDebug("RUNMGR->callForward() : %d layers will be executed", _index[0]);
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);
    int64_t inputs_L = _tmp_outputs;
    int8_t ret = forward_network_ca(invitation, _input, (inputs_L * _batch), _index);
    return 0;
}
void network::fetchOutput_CA(TEEC_INVITATION_T *TEEC_INVITATION) {
    LogDebug("RUNMGR->fetchOutput()");
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);

    Layer *ll = getLastL();
    int64_t outputs_L = (nullptr == ll ? _tmp_outputs : ll->getOutputs());

    int8_t ret = forward_ret_network_ca(invitation, _output, (outputs_L * _batch));
}
void network::makeNet_CA(TEEC_INVITATION_T *TEEC_INVITATION) {
    LogDebug("RUNMGR->makeNet_CA() : Layers number = %d", _num);
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);
    make_network_ca(invitation,
                    Int64TToINTTA(_num),
                    Int64TToINTTA(_batch),
                    Int64TToINTTA(_inputs),
                    Int64TToINTTA(_outputs),
                    Int64TToINTTA(_inoutSize),
                    Int64TToINTTA(_workspace_size),
                    _clip,
                    _train);
    
    Layer *l = NULL;
    for (size_t i = 0; i < _Layers.size(); ++i) {
        l = _Layers.at(i);
        // 打印函数地址
        l->make_layer_CA(invitation);
        l->make_layer_ext_CA(invitation);

    }
}
void network::makeNet_Pthreads_CA() {
    LogDebug("RUNMGR->makeNet_Pthreads_CA() : Layers number = %d", _num);
    
    // 这里需要在各个线程中初始化 net 
    // ......
}
#endif

// =============================================================================
#if LIBTORCH_EXT
void paramShift(const std::string &str, const at::Tensor &data, int64_t idx) {
    // 这里的 idx 本质上仅仅代表需要进行权重转移的 Layer 的下标
    // Layer *player = NULL;
    // int64_t idx_l = 0;
    // while (idx >= 0) {
    //     player = NETWORK->getLayer(idx_l);
    //     if (isParams(player->getType())) --idx;
    //     ++idx_l;
    // }
    
    Layer *player = NETWORK->getLayer(Int64TToInt32T(idx));

    EXIT_ERROR_CHECK(player, nullptr, "Cannot find the corresponding layer through the idx");
    // player->paramLoad(str, data);
    EXIT_ERROR_CHECK(data.dtype() == at::kFloat, false, "type in data if not FLOATCA");
    if (!strcmp(str.c_str(), "weight")) {player->setWeights(data);}
    else if (!strcmp(str.c_str(), "bias")) {player->setBiases(data);} 
    // ......todo
    else if (!strcmp(str.c_str(), "inputs")) {player->initialize(data);}
    else if (!strcmp(str.c_str(), "mean")) {player->setMeans(data);}
    else if (!strcmp(str.c_str(), "variance")) {player->setVariances(data);}
}
#endif // LIBTORCH_EXT

#if LIBTORCH_EXT
at::Tensor getTensor() {
// #if TORCHZONE
if (Flag_TorchZone) {
#if TORCHZONE
    NETWORK->callForward_CA();
    NETWORK->fetchOutput_CA();
    // NETWORK->Forward_Fetch_CA();

    Layer *ll = NETWORK->getLastL();
    int64_t const * const outdims = (nullptr == ll ?  NETWORK->getOutDmis_t() : ll->getOutDmis_t());

    std::vector<int64_t> sizes;
    sizes.push_back(NETWORK->getBatch());
    for (size_t i = 1; i <= outdims[0]; ++i) {sizes.push_back(outdims[i]);}

    // for (int i = 0; i < l->getOutputs(); ++i) {std::cout << *(NETWORK->getOutput() + i) << " ";} 
    // std::cout << std::endl;

    // 这里的张量通过浅拷贝方式返回 data 生命周期通过 net 管理
    return makeTensor(NETWORK->getOutput(), sizes);
// #else 
#endif
}
    // std::cout << "getTensor : TORCHZONE not define : return empty" << std::endl;
    // return at::empty({});

    NETWORK->callForward();

    FLOATCA *NET_OUTPUT = NETWORK->getOutput();

    // 在多核心并行的情况下 输入输出维度的管理比较复杂
    // 下面这个理论上是有 bug 存在的
    // 后续最好的解决办法就是重构一下 把输入输出与 net 和 layer 解耦 ...... todo
    Layer *ll = NETWORK->getLastL();
    int64_t const * const outdims = (nullptr == ll ?  NETWORK->getOutDmis_t() : ll->getOutDmis_t());

    std::vector<int64_t> sizes;
    sizes.push_back(NETWORK->getBatch());
    for (size_t i = 1; i <= outdims[0]; ++i) {sizes.push_back(outdims[i]);}

    // for (int i = 0; i < l->getOutputs(); ++i) {std::cout << *(NETWORK->getOutput() + i) << " ";} 
    // std::cout << std::endl;
    
    // ==============================
    int64_t outputs = NETWORK->getPreOutputs();
    FLOATCA *NET_INPUT = NETWORK->getInput();
    memcpy(NET_INPUT, NET_OUTPUT, outputs * 4);
    // ==============================

    // 这里的张量通过浅拷贝方式返回 data 生命周期通过 net 管理
    return makeTensor(NET_OUTPUT, sizes);

    // printf("output : %p\n", NETWORK->getOutput());
    // printf("Tensor : %p\n", out.data_ptr<FLOATCA>());
// #endif

}
void resetNet(at::Tensor &input) {
    LogDebug("void resetNet(at::Tensor &)");
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
}
void network::partion_Stn() {
    std::stable_partition(_Layers.begin(), _Layers.end(), 
    [](const Layer *l){return l->getType() != TSTATION_TYPE;});
    resetLayersIndex();

    Layer *l = nullptr;
    for (int64_t i = 0; i < _num; ++i) {
        l = _Layers[i];
        if (TSTATION_TYPE == l->getType()) l->initialize(at::empty({}));
    }
}
void partionLayer() {NETWORK->partion_Stn();}
#endif // LIBTORCH_EXT

void setTorchZone(int8_t flag) {
    Flag_TorchZone = flag;
}
void HeapAllocTrack() {NETWORK->memUsage_heap();}

} // namespace end of core
} // namespace end of PyTZone 
