#include "./tensor.h"
#if LIBTORCH_EXT
namespace PyTZone {

at::Tensor makeTensor(FLOATCA *data, const std::vector<int64_t> &sizes) {
    std::function<void(void*)> deleter = [](void*){};
    return at::from_blob(data, sizes, deleter, torch::TensorOptions().dtype(torch::kFloat));
    // return at::empty({});
}

} // namespace end of PyTZone
#endif // LIBTORCH_EXT