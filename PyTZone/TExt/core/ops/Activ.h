#ifndef __ACTIV_H__
#define __ACTIV_H__

#include <core/Layer.h>
#include <core/network.h>
#include <unordered_map>

using std::string;
using std::unordered_map;

// def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
//     super().__init__()
//     self.negative_slope = negative_slope
//     self.inplace = inplace

namespace PyTZone {
namespace core {

class Activ: 
#if LIBTORCH_EXT
public torch::CustomClassHolder,
#endif // LIBTORCH_EXT
virtual public Layer
{
public:
    ~Activ();
    Activ() = delete;
    Activ(int64_t activ);
    Activ(std::string activ);

    void print() const override;
    void forward() override;
    void setActivate();

#if TORCHZONE
    void make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr) const override;
#endif
#if LIBTORCH_EXT
    Activ(SerActivPrePack spp);
    void paramLoad(const string &str, const at::Tensor &data) override;
    void initialize(const at::Tensor &data) override;
    const SerActivPrePack &makePack();
    void operator()(at::Tensor &input);
    void operator()();
    void attach(at::Tensor &input);
    void attach();
#endif // LIBTORCH_EXT
private:
    // negative_slope: float = 1e-2, inplace: bool = False
    // double _negative_slope;
    // int8_t _inplace;

    ACTIVATION _activ;
};


} // namespace end of core
} // namespace end of PyTZone

#endif