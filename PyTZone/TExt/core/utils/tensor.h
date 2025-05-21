#ifndef __TENSOR_H__
#define __TENSOR_H__

#if LIBTORCH_EXT
#include <RunMgr/all.h>
namespace PyTZone {

// ==============================================================================
    /* const std::string &str, const at::Tensor &data */

    /* auto* data_ptr = data.data_ptr<FLOATCA>(); */

    /* conv1.weight 或 conv1.bias */
    /* cout << str.c_str() << endl; */

    /* // 查看维度 */
    /* // num : channel : kernel_size : kernel_size */
    /* // data.size() = [64, 32, 3, 3]    :    data.numel() = 18432 */
    /* cout << "data.size() = ["; */
    /* for (int i = 0; i < data.dim(); ++i) {  // int64_t */
    /*     cout << data.size(i);   // int64_t */
    /*     if (i < data.dim() - 1) cout << ", "; */
    /* } */
    /* cout << "]    :    " << "data.numel() = " << data.numel() << endl;  // int64_t */

    /* cout << *(data_ptr + 27) << endl; */
// ==============================================================================

// at::Tensor makeTensor(FLOATCA *data, int64_t *size);
at::Tensor makeTensor(FLOATCA *data, const std::vector<int64_t> &sizes); 

} // namespace end of PyTZone

#endif // LIBTORCH_EXT
#endif
