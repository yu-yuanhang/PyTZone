#ifndef __NormBatch_H__
#define __NormBatch_H__

#include <core/Layer.h>
#include <core/network.h>
#include <random> 
// =========================================================
// class _BatchNorm(_NormBase):
//     def __init__(
//         self,
//         num_features: int,               // 这是输入张量中每个特征的数量
//         eps: float = 1e-5,               // 在计算标准差时添加的小常数 避免出现除以零的情况
//         momentum: Optional[float] = 0.1, // 动量用于更新均值和方差的移动平均值
//         affine: bool = True,             // true 则 BatchNorm 层会学习可训练的缩放 gamma 和偏移 beta 参数
//         track_running_stats: bool = True,    // 控制是否追踪每个特征的运行时均值和方差 以便在评估或测试期间使用
//         device=None,
//         dtype=None,
//     ) -> None:
// =========================================================
// class _InstanceNorm(_NormBase):
//     def __init__(
//         self,
//         num_features: int,
//         eps: float = 1e-5,
//         momentum: float = 0.1,
//         affine: bool = False,
//         track_running_stats: bool = False,
//         device=None,
//         dtype=None,
//     ) -> None:
// =========================================================
// class _LazyNormBase(LazyModuleMixin, _NormBase):
//     weight: UninitializedParameter  # type: ignore[assignment]
//     bias: UninitializedParameter  # type: ignore[assignment]

//     def __init__(
//         self,
//         eps=1e-5,
//         momentum=0.1,
//         affine=True,
//         track_running_stats=True,
//         device=None,
//         dtype=None,
//     ) -> None:
// =========================================================
// class _LazyNormBase(LazyModuleMixin, _NormBase):
//     weight: UninitializedParameter  # type: ignore[assignment]
//     bias: UninitializedParameter  # type: ignore[assignment]

//     def __init__(
//         self,
//         eps=1e-5,
//         momentum=0.1,
//         affine=True,
//         track_running_stats=True,
//         device=None,
//         dtype=None,
//     ) -> None:
// =========================================================

namespace PyTZone {
namespace core {


class NormNd:
#if LIBTORCH_EXT
public torch::CustomClassHolder,
#endif // LIBTORCH_EXT
virtual public Layer 
{
public:
    void print() const override;
    void updateNet_thr(List<struct subNet_s> &subNet, size_t threadSize) override;
    void memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const;

#if LIBTORCH_EXT
    const SerNormNdPrePack &makePack(); 
    void paramLoad(const string &str, const at::Tensor &data) override;
    void initialize(const at::Tensor &data) override;
#endif // LIBTORCH_EXT
#if TORCHZONE
    void make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr) const override;
#endif
protected:
#if LIBTORCH_EXT
    NormNd(SerNormNdPrePack &&spp);
#endif // LIBTORCH_EXT
    NormNd() = delete;
    ~NormNd();
    NormNd(int64_t in_c = INVALID_VALUE_U,   // 取决于 Norm 的类型
           double eps = 1e-5,
           double momentum = 0.1,
           int8_t affine = true,    // 理论上将会影响 weights 和 biases 是否存在
           int8_t track_running_stats = false
           /* device */
           /* dtype */);

protected:
    // torchscript 接口不支持 float
    int64_t _in_c;
    double _eps;
    double _momentum;
    int8_t _affine;
    int8_t _track_running_stats;
    // device=None,
    // dtype=None,
};

// ================================================================
// BatchNorm 
class BatchNorm2d:
public NormNd {
public:
    BatchNorm2d() = delete;
    ~BatchNorm2d();
#if LIBTORCH_EXT
    BatchNorm2d(SerNormNdPrePack spp);
#endif // LIBTORCH_EXT

    BatchNorm2d(int64_t in_c,
                double eps = 1e-5,
                double momentum = 0.1,
                int8_t affine = false,
                int8_t track_running_stats = false
                /* device */
                /* dtype */);

#if LIBTORCH_EXT
    void operator()(at::Tensor &input);
    void operator()();
    // 作用 同函数调用运算符
    void attach(at::Tensor &input);
    void attach();
#endif // LIBTORCH_EXT

    void forward() override;

};


} // namespace end of core
} // namespace end of PyTZone

#endif