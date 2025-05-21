#ifndef __SERIALIZER_H__
#define __SERIALIZER_H__

#include <core/ops/Conv.h>

// using SerializationTypeConv2dPrePack = std::tuple<
//     Tensor,
//     std::optional<Tensor>,
//     std::vector<int64_t>,
//     std::vector<int64_t>,
//     std::vector<int64_t>,
//     int64_t,
//     std::optional<Scalar>,
//     std::optional<Scalar>>;

namespace PyTZone {

#if SERIALIZER

using namespace core;

// 用来为所有的类进行序列化和反序列化
// 每一个 Context 都需要设置为 ops 非顶层算子的友元
class ConvOpsCtx:
public torch::CustomClassHolder {
public:
    SerConvNdPrePack pack() const;
    // ConvOpsCtx() = delete;
    ConvOpsCtx() {};
    ConvOpsCtx(ConvNd conv);
    // ConvOpsCtx(ConvNd conv);
    ConvOpsCtx(
        std::vector<int64_t> params,
        at::Tensor weights,
        std::optional<at::Tensor> bias,
        std::vector<int64_t> dims,
        std::vector<int64_t> out_dims,
        int8_t binary,
        int8_t xnor,
        int64_t channel,
        int64_t num,
        std::vector<int64_t> size,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        int8_t isBias,
        int64_t padding_mode,
        int64_t dimsNum): 
        _params(std::move(params)),
        _weights(std::move(weights)),
        _bias(std::move(bias)),
        _dims(std::move(dims)),
        _out_dims(std::move(out_dims)),
        _binary(binary),
        _xnor(xnor),
        _channel(channel),
        _num(num),
        _size(std::move(size)),
        _stride(std::move(stride)),
        _padding(std::move(padding)),
        _dilation(std::move(dilation)),
        _groups(groups),
        _isBias(isBias),
        _padding_mode(padding_mode),
        _dimsNum(dimsNum) {}

    inline const std::vector<int64_t> &get_params() const {
        return _params;
    }
    inline const at::Tensor &get_weights() const {
        return _weights;
    }
    inline bool isBias() const {
        return _bias.has_value();
    }
    // 理论上该函数具有风险 需要 isBias 配合使用
    inline const std::optional<at::Tensor> &get_bias() const {
        return _bias;
    }
    inline const std::vector<int64_t> &get_dims() const {
        return _dims;
    }
    inline const std::vector<int64_t> &get_out_dims() const {
        return _out_dims;
    }
    inline int8_t get_binary() const {
        return _binary;
    }
    inline int8_t get_xnor() const {
        return _xnor;
    }
    inline int64_t get_channel() const {
        return _channel;
    }
    inline int64_t get_num() const {
        return _num;
    }
    inline const std::vector<int64_t> &get_size() const {
        return _size;
    }
    inline const std::vector<int64_t> &get_stride() const {
        return _stride;
    }
    inline const std::vector<int64_t> &get_padding() const {
        return _padding;
    }
    inline const std::vector<int64_t> &get_dilation() const {
        return _dilation;
    }
    inline int64_t get_groups() const {
        return _groups;
    }
    inline int8_t get_isBias() const {
        return _isBias;
    }
    inline int64_t get_padding_mode() const {
        return _padding_mode;
    }
    inline int64_t get_dimsNum() const {
        return _dimsNum;
    }
private:
// using SerPrePack_Layer = std::tuple<
//     std::vector<int64_t>,  
//     at::Tensor,
//     std::optional<at::Tensor>>;

// using SerConvNdPrePack = std::tuple<
//     SerPrePack_Layer,
//     std::vector<int64_t>,   // dims
//     std::vector<int64_t>,   // out_dims
//     std::tuple<int8_t, int8_t>,   
//     std::tuple<int64_t, int64_t>,
//     std::vector<int64_t>,   
//     std::vector<int64_t>,  
//     std::vector<int64_t>,   
//     std::vector<int64_t>, 
//     int64_t, int8_t, int64_t, int64_t>;

    // Layer
    std::vector<int64_t> _params;
    at::Tensor _weights;
    std::optional<at::Tensor> _bias;
    // ConvNd

    std::vector<int64_t> _dims;
    std::vector<int64_t> _out_dims;
    int8_t _binary;
    int8_t _xnor;
    int64_t _channel;
    int64_t _num;
    std::vector<int64_t> _size;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _padding;
    std::vector<int64_t> _dilation;
    int64_t _groups;
    int8_t _isBias;
    int64_t _padding_mode;
    int64_t _dimsNum;
};
#endif  // SERIALIZER

} // namespace end of PyTZone

#endif