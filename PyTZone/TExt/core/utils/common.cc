#include "./common.h"

namespace PyTZone {

tuple<string, string> splitString_dot(const string &str) {
    size_t pos = str.find('.');
    // 用来检查 find 方法是否找到了指定的子字符串
    if (pos != std::string::npos) {
        std::string firstPart = str.substr(0, pos);
        std::string secondPart = str.substr(pos + 1);
        return std::make_tuple(firstPart, secondPart);
    }
    return std::make_tuple(str, "NULL"); // 如果没有找到，返回原字符串和空字符串
}
int64_t getDimsIdx(int64_t *dims, int64_t idx) {
    RET_ERROR_CHECK(dims, nullptr, "mullptr == dims", SIZE_MAX);
    RET_ERROR_CHECK(idx < 0, true, "idx < 0", SIZE_MAX);
    RET_ERROR_CHECK(idx > *dims, true, "idx is too long ", SIZE_MAX);
    return *(dims + idx);
}

size_t int64ToSizeT(int64_t value) {
    RET_ERROR_CHECK(value < 0, true, "value < 0", SIZE_MAX);
    return static_cast<size_t>(value);
}
size_t int8ToSizeT(int8_t value) {
    RET_ERROR_CHECK(value < 0, true, "value < 0", SIZE_MAX);
    return static_cast<size_t>(value);
}
int64_t SizeTToInt64(size_t value) {
    // if (value > static_cast<size_t>(INT64_MAX)) {
    //     throw std::overflow_error("size_t value is too large to convert to int64_t.");
    // }
    RET_ERROR_CHECK(value > static_cast<size_t>(INT64_MAX), true, "size_t value is too large", INT64_MAX);
    return static_cast<int64_t>(value);
}
int8_t SizeTToInt8(size_t value) {
    // if (value > static_cast<size_t>(INT8_MAX)) {
    //     throw std::overflow_error("size_t value is too large to convert to int8_t.");
    // }
    RET_ERROR_CHECK(value > static_cast<size_t>(INT8_MAX), true, "size_t value is too large", INT8_MAX);
    return static_cast<int8_t>(value);
}
int32_t Int64TToInt32T(int64_t value) {
    if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
        RET_ERROR_CHECK(1, 1, "value is too long ", INT32_MAX);
    return static_cast<int32_t>(value);
}
#if TORCHZONE
INT_TA Int64TToINTTA(int64_t value) {
    if (sizeof(int64_t) > sizeof(INT_TA)) {
        if (value < std::numeric_limits<INT_TA>::min() || value > std::numeric_limits<INT_TA>::max())
            RET_ERROR_CHECK(1, 1, "value is too long ", 0);
        return static_cast<INT_TA>(value);
    } else return static_cast<INT_TA>(value);
}
INT_TA Int64TToINTTA(int32_t value) {
    if (sizeof(int32_t) > sizeof(INT_TA)) {
        if (value < std::numeric_limits<INT_TA>::min() || value > std::numeric_limits<INT_TA>::max())
            RET_ERROR_CHECK(1, 1, "value is too long ", 0);
        return static_cast<INT_TA>(value);
    } else return static_cast<INT_TA>(value);
}
#endif


// uint32_t int64ToUint32T(int64_t value) {
//     RET_ERROR_CHECK(value < 0, true, "value < 0", UINT32_MAX);
//     RET_ERROR_CHECK(value > UINT32_MAX, true, "value > UINT32_MAX", UINT32_MAX);
//     return static_cast<uint32_t>(value);
// }
// uint32_t SizeTToUint32T(size_t value) {
//     RET_ERROR_CHECK(value > UINT32_MAX, true, "value > UINT32_MAX", UINT32_MAX);
//     return static_cast<uint32_t>(value);
// }
// std::vector<int64_t> SizeToInt64Vec(const std::vector<size_t> &vec) {
//     std::vector<int64_t> int64_tVector(vec.size());
//     std::transform(vec.begin(), vec.end(), int64_tVector.begin(),
//                 [](size_t value) {
//                     RET_ERROR_CHECK(value > static_cast<size_t>(INT64_MAX), true, "size_t value is too large", INT64_MAX);
//                     return static_cast<int64_t>(value);});
//     return int64_tVector;
// }
// std::vector<size_t> Int64ToSizeVec(const std::vector<int64_t> &vec) {
//     std::vector<size_t> size_tVector(vec.size());
//     std::transform(vec.begin(), vec.end(), size_tVector.begin(),
//                 [](int64_t value) {
//                     RET_ERROR_CHECK(value < 0, true, "error value in std::vector<int64_t>", SIZE_MAX);
//                     return static_cast<size_t>(value);});
//     return size_tVector;
// }

} // namespace end of PyTZone