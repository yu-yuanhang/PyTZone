#ifndef __SERPREPACK_H__
#define __SERPREPACK_H__

#include <RunMgr/all.h>

namespace PyTZone {

#define SERIALIZER 0

// 这里作为友元类的前向声明(尽管这里在同一个上下文中友元类可以省去前向声明)
// 前向声明类时不能包括其继承关系的具体细节
class ConvOpsCtx;

enum LAYER_M_IDX {
    TYPE = 0,
    DIMSNUM,
    BINARY,
    XNOR,
    KEEPIN,
    KEEPOUT,
    BATCH,
    INPUTS,
    OUTPUTS,
    NWEIGHTS,
    NBIASES,
    WORKSPACE_SIZE,
    IDX,
};

#if SERIALIZER
// 这里将类型转换的事情交给 Layers
int64_t getForLayer(const std::vector<int64_t> &vec, LAYER_M_IDX idx) {
    EXIT_ERROR_CHECK(vec.size() <= static_cast<size_t>(idx), true, "error params from ConvOpsCtx");
    return vec.at(idx);
}
#endif // SERIALIZER

#if LIBTORCH_EXT
using SerPrePack_Layer = std::tuple<
    std::vector<int64_t>,  
    std::vector<int64_t>,   // dims
    std::vector<int64_t>,   // out_dims
    at::Tensor,             
    std::optional<at::Tensor>,
    at::Tensor,             // 均值方差 用于归一化
    at::Tensor>;

using SerConvNdPrePack = std::tuple<
    SerPrePack_Layer,
    int64_t, int64_t,
    std::vector<int64_t>,   
    std::vector<int64_t>,  
    std::vector<int64_t>,   
    std::vector<int64_t>, 
    int64_t, int8_t, int64_t>;

using SerActivPrePack = std::tuple<SerPrePack_Layer, int64_t>;

using SerPoolPrePack = std::tuple<
    SerPrePack_Layer,
    std::vector<int64_t>,   
    std::vector<int64_t>,  
    std::vector<int64_t>,   
    std::vector<int64_t>,
    int8_t, int8_t, int8_t,
    int64_t, 
    int64_t>;

using SerLinearPrePack = std::tuple<
    SerPrePack_Layer,
    int64_t, int64_t, int8_t>;

using SerNormNdPrePack = std::tuple<
    SerPrePack_Layer,
    int64_t,
    double, double,
    int8_t, int8_t>;

using SerTStationPrePack = std::tuple<
    SerPrePack_Layer, 
    int64_t, 
    int64_t, int8_t, 
    int64_t, int8_t>;

#endif // LIBTORCH_EXT

} // namespace end of PyTZone


#endif