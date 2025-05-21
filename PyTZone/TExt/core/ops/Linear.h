#ifndef __LINEAR_H__
#define __LINEAR_H__

#include <core/Layer.h>
#include <core/network.h>
#include <random> 

// def __init__(
//     self,
//     in_features: int,
//     out_features: int,
//     bias: bool = True,
//     device=None,
//     dtype=None,
// ) -> None:

namespace PyTZone {
namespace core {

class Linear:
#if LIBTORCH_EXT
public torch::CustomClassHolder,
#endif // LIBTORCH_EXT
virtual public Layer {
public:

    void forward() override;

    void print() const override;
    inline void setKeep(int8_t keepIn, int8_t keepOut) {
        LogDebug("void Linear::setKeep(int8_t, int8_t) : idx = %d : ", _idx);
        _keepIn = keepIn;
        _keepOut = keepOut;
        // return this;
    }
    inline int64_t getIdx() const {return this->_idx;}

    void updateNet_thr(List<struct subNet_s> &subNet, size_t threadSize) override;
    void memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const;

#if LIBTORCH_EXT
    Linear(SerLinearPrePack spp);
    const SerLinearPrePack &makePack(); 
    void paramLoad(const string &str, const at::Tensor &data) override;
    void initialize(const at::Tensor &data) override;

    void operator()(at::Tensor &input);
    void operator()();
    // 作用 同函数调用运算符
    void attach(at::Tensor &input);
    void attach();
#endif // LIBTORCH_EXT
#if TORCHZONE
    void make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr) const override;
#endif

    Linear() = delete;
    ~Linear();
    Linear(int64_t in_c, int64_t out_c, int8_t isBias = true);

private:
    /* int8_t _adam;              // 是否使用 adam 优化器 */
    int64_t _in_c;
    int64_t _out_c;
    int8_t _isBias;             // 默认 true
    // device=None,
    // dtype=None,
};


} // namespace end of core
} // namespace end of PyTZone

#endif