#include "Serializer.h"

namespace PyTZone {

#if SERIALIZER

using namespace core;

SerConvNdPrePack ConvOpsCtx::pack() const {
    return std::make_tuple(
        std::make_tuple(std::move(_params), std::move(_weights), std::move(_bias)),
        std::move(_dims),
        std::move(_out_dims),
        _binary,
        _xnor,
        _channel,
        _num,
        std::move(_size),
        std::move(_stride),
        std::move(_padding),
        std::move(_dilation),
        _groups,
        _isBias,
        _padding_mode,
        _dimsNum);
}

ConvOpsCtx::ConvOpsCtx(ConvNd conv):
    _params(std::move(conv.makeParams())),
    _weights(std::move(conv.getWeights())),
    _bias(std::move(std::optional<at::Tensor>(std::move(std::move(conv.getBiases()))))),
    _dims(arrayToVector<int64_t>((conv._dims + 1), conv._dims[0])),
    _out_dims(arrayToVector<int64_t>((conv._out_dims + 1), conv._out_dims[0])),
    _binary(conv._binary),
    _xnor(conv._xnor),
    _channel(conv._channel),
    _num(conv._num),
    _size(conv._size.getVector()),
    _stride(conv._stride.getVector()),
    _padding(conv._padding.getVector()),
    _dilation(conv._dilation.getVector()),
    _groups(conv._groups),
    _isBias(conv._isBias),
    _padding_mode(static_cast<int64_t>(conv._padding_mode)),
    _dimsNum(static_cast<int64_t>(conv._dimsNum)) {}
// ConvOpsCtx::ConvOpsCtx(ConvNd &&conv):
//     _params(std::move(conv.makeParams())),
//     _weights(std::move(conv.getWeights())),
//     _bias(std::move(std::optional<at::Tensor>(std::move(std::move(conv.getBiases()))))),
//     _dims(arrayToVector<int64_t>((conv._dims + 1), conv._dims[0])),
//     _out_dims(arrayToVector<int64_t>((conv._out_dims + 1), conv._out_dims[0])),
//     _binary(conv._binary),
//     _xnor(conv._xnor),
//     _channel(conv._channel),
//     _num(conv._num),
//     _size(conv._size.getVector()),
//     _stride(conv._stride.getVector()),
//     _padding(conv._padding.getVector()),
//     _dilation(conv._dilation.getVector()),
//     _groups(conv._groups),
//     _isBias(conv._isBias),
//     _padding_mode(static_cast<int64_t>(conv._padding_mode)),
//     _dimsNum(static_cast<int64_t>(conv._dimsNum)) {}

#endif  // SERIALIZER

} // namespace end of PyTZone
