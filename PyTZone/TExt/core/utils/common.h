#ifndef __COMMON_H__
#define __COMMON_H__

#include <RunMgr/all.h>
#include <vector>
#include <algorithm> 
#include <limits>
#include <stdexcept>

using std::string;
using std::tuple;

namespace PyTZone {

tuple<string, string> splitString_dot(const string &str); 

size_t int64ToSizeT(int64_t value);
size_t int8ToSizeT(int8_t value);

int64_t getDimsIdx(int64_t *dims, int64_t idx);

// template <typename T, int64_t N>
// std::vector<T> arrayToVector(const T (&arr)[N]) {
//     return std::vector<T>(std::begin(arr), std::end(arr));
// }
template <typename T>
std::vector<T> arrayToVector(const T *arr, size_t N) {
    std::vector<T> vec;
    vec.reserve(N);
    for(size_t i = 0; i < N; ++i) {
        vec.push_back(*(arr + i));
    }
    return vec;
}
int64_t SizeTToInt64(size_t value);
int8_t SizeTToInt8(size_t value);
int32_t Int64TToInt32T(int64_t value);

#if TORCHZONE
INT_TA Int64TToINTTA(int64_t value);
INT_TA Int32TToINTTA(int64_t value);
#endif 
// ======================================================
// 框架的底层数据类型修改后 这些函数目前无用
// uint32_t int64ToUint32T(int64_t value);
// uint32_t SizeTToUint32T(size_t value);
// std::vector<int64_t> SizeToInt64Vec(const std::vector<size_t> &vec);
// std::vector<size_t> Int64ToSizeVec(const std::vector<int64_t> &vec);


} // namespace end of PyTZone
#endif
