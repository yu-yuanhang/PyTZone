#ifndef __TSOPS_H__
#define __TSOPS_H__

#include <core/Layer.h>
#include <core/network.h>
#include <unordered_map>

using std::string;

namespace PyTZone {
namespace core {

class TStation: 
#if LIBTORCH_EXT
public torch::CustomClassHolder,
#endif // LIBTORCH_EXT
virtual public Layer
{
public:
    ~TStation();
    TStation() = delete;
    TStation(int64_t stn, 
             int64_t index2, int8_t _kpIdx2,
             int64_t index1 = INVALID_VALUE,
             int8_t _kpIdx1 = INVALID_VALUE_U);

    void print() const override;
    void memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const override {}
    inline void subIndex() {
        if (_index1 >= 0 && nullptr != _layer1) 
            _index1 = _layer1->getIdx();
        if (_index2 >= 0 && nullptr != _layer2) 
            _index2 = _layer2->getIdx();
    }
    inline void setKeep(int8_t keepIn, int8_t keepOut) {
        LogDebug("void TStation::setKeep(int8_t, int8_t) : idx = %d : ", _idx);
        if (TSTATION_TYPE > _type && 100 < _type) {
            if (keepIn && !_keepIn) {_input = new FLOATCA[_inputs];}
            if (!keepIn && _keepIn) {delete [] _input;_input = nullptr;}
            if (keepOut && !_keepOut) {_output = new FLOATCA[_outputs];}
            if (!keepOut && _keepOut) {delete [] _output;_output = nullptr;}
        }
        _keepIn = keepIn;
        _keepOut = keepOut;
        // return this;
    }
    inline int64_t getIdx() const {return this->_idx;}
#if TORCHZONE
    void make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr) const override;
#endif
#if LIBTORCH_EXT
    TStation(SerTStationPrePack spp);
    void paramLoad(const string &str, const at::Tensor &data) override;
    void initialize(const at::Tensor &data) override;
    const SerTStationPrePack &makePack();
    void operator()(at::Tensor &input);
    void operator()();
    void attach(at::Tensor &input);
    void attach();
#endif // LIBTORCH_EXT

    void forward() override;
    void setLayerPtr() {
        _layer1 = (_index1 >= 0) ? _layer1 = NETWORK->getLayer(_index1) : nullptr;
        _layer2 = (_index2 >= 0) ? _layer2 = NETWORK->getLayer(_index2) : nullptr;
    }

private:
    STATION _stn;
    int64_t _index2;    // index for Layer keep
    int8_t _kpIdx2;      // true ==> input

    // 这里从设计逻辑上来讲 _index1 应该仅代表前一层网络
    // 但是有些时候无法设置所以目前大多数时候被设置成空值
    int64_t _index1;    // inputs net
    int8_t _kpIdx1;
};


} // namespace end of core
} // namespace end of PyTZone

#endif