#ifndef __CUSTOMARRAY_H__
#define __CUSTOMARRAY_H__

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <tuple>

#include <Logger/logger.h>
#include "Singleton.h"

using std::array;
using std::vector;
using std::string;

namespace PyTZone {

#if 1
template <typename T, std::size_t N, T DefVal>
class CustomArray_def {
public:
    /* CustomArray_def() {} */
    /* ~CustomArray_def() {} */

    // 在编译期间验证某些条件是否成立
    static_assert(std::is_same<decltype(DefVal), T>::value, "DefVal must be of type T!");

    CustomArray_def(const vector<T> &vec = vector<T>()): _n(0) {
        if (vec.size() > N)
            LogWarn("CustomArray_def(const vector<T> &) vec.size() > N : %zu", vec.size());    
        
        size_t idx = 0;
        for (T val : vec) {
            if(idx < N) _arr[idx++] = val;
        }
        // 这里添加 默认值 
        // 0 == vec.size() 等价于 0 == idx
        if (0 == vec.size()) _arr[idx++] = static_cast<T>(DefVal);
       
        // std::fill(_arr.begin() + idx, _arr.end(), _arr[idx - 1]);
        std::size_t tmp = idx - 1;
        for (idx; idx < N; ++idx) _arr[idx] = _arr[tmp];
    }
    
    template <typename ...Args>
    CustomArray_def(const std::tuple<Args...> &tup = std::make_tuple(static_cast<T>(DefVal))): _n(0)  {
        size_t tupleSize = std::tuple_size<std::decay_t<decltype(tup)>>::value;
        if (tupleSize > N)
            LogWarn("CustomArray_def(const vector<T> &) vec.size() > N : %zu", tupleSize);    
        if (!checkTupleTypes(tup)) {
            LogWarn("error parameter type in tuple");    
            exit(EXIT_FAILURE);
        }
        // EXIT_ERROR_CHECK(checkTupleTypes<T>(tup), false, "error parameter type in tuple");

        // 以下开始 赋值 但是其实 对于顶层算子的构造函数而言 
        // 因为 pybind11 不能直接推导模板参数 所以 目前只能靠重载实现 
        // 暂时 不会出现 元组不确定大小的情况
        assignValues(tup);
        // size_t idx = 0;
        // if (0 == tupleSize) _arr[idx++] = static_cast<T>(DefVal);
        // std::fill(_arr.begin() + idx, _arr.end(), _arr[idx - 1]);
    }

    CustomArray_def(const CustomArray_def &rhs):_n(rhs.getn()) {
        for(size_t i = 0; i < N; ++i) _arr[i] = rhs.getData(i);
    }

    // 检查元组中每个元素的类型是否为 T
    template <typename Tuple, std::size_t... Indices>
    bool checkTupleTypes(const Tuple& tup, std::index_sequence<Indices...>) {
        return ((std::is_same<std::decay_t<decltype(std::get<Indices>(tup))>, T>::value) && ...);
    }

    // 主函数，获取元组大小并创建索引序列
    template <typename Tuple>
    bool checkTupleTypes(const Tuple& tup) {
        constexpr std::size_t size = std::tuple_size<Tuple>::value;
        return checkTupleTypes(tup, std::make_index_sequence<size>{});
    }

    template <typename Tuple, std::size_t... Indices>
    void assignValues(const Tuple& tup, std::index_sequence<Indices...>) {
        size_t idx = 0;
        // 使用索引序列遍历元组元素
        ((idx < N ? (_arr[idx++] = std::get<Indices>(tup), void()) : void()), ...);
    }

    template <typename Tuple>
    void assignValues(const Tuple& tup) {
        constexpr std::size_t size = std::tuple_size<Tuple>::value;
        assignValues(tup, std::make_index_sequence<size>{});
    }

    T &operator[](size_t idx) {return _arr[idx];}

    void print() const {
        std::cout << "n = " << _n << " : arr = ";
        for (std::size_t i = 0; i < _n; ++i) std::cout << _arr[i] << " ";
        std::cout << std::endl; 
    }

    T getData(std::size_t idx) const {
        return _arr[idx];
    }
    std::size_t getn() const {
        return _n;
    }
    std::vector<T> getVector() const {
        return std::vector<T>(_arr, _arr + _n);
    }

    void truncateArr(std::size_t n) {
        _n = n;
        for (std::size_t i = _n; i < N; ++i) _arr[i] = 0;
    }
// public:
private:
    std::size_t _n;
    // array<T, N> _arr;
    T _arr[N];
};

template <typename T, std::size_t N>
class CustomArray {
public:
    /* CustomArray() {} */
    /* ~CustomArray() {} */

    CustomArray(const vector<T> &vec = vector<T>()):_n(0) {
        if (vec.size() > N)
            LogWarn("CustomArray_def(const vector<T> &) vec.size() > N : %zu", vec.size());  
        else if (0 == vec.size())
            LogWarn("CustomArray_def(const vector<T> &) 0 == vec.size() : %zu", vec.size()); 

        size_t idx = 0;
        for (T val : vec) {
            if(idx < N) _arr[idx++] = val;
        }
        // 这里添加 默认值 
        // 0 == vec.size() 等价于 0 == idx
        // std::fill(_arr.begin() + idx, _arr.end(), _arr[idx - 1]);

        // vec.size() >= 0
        std::size_t tmp = idx - 1;
        for (idx; idx < N; ++idx) _arr[idx] = _arr[tmp];
    }
    // 拷贝构造函数是危险的
    // 必须先保证两个类型的 N 相同
    CustomArray(const CustomArray &rhs):_n(rhs.getn()) {
        for(size_t i = 0; i < N; ++i) _arr[i] = rhs.getData(i);
    }

    T &operator[](size_t idx) {return _arr[idx];}

    void print() const {
        std::cout << "n = " << _n << " : arr = ";
        for (std::size_t i = 0; i < _n; ++i) std::cout << _arr[i] << " ";
        std::cout << std::endl; 
    }

    T getData(std::size_t idx) const {
        return _arr[idx];
    }
    std::size_t getn() const {
        return _n;
    }
    std::vector<T> getVector() const {
        return std::vector<T>(_arr, _arr + _n);
    }
    void truncateArr(std::size_t n) {
        _n = n;
        for (std::size_t i = _n; i < N; ++i) _arr[i] = 0;
    }
// public:
private:
    std::size_t _n;
    // array<T, N> _arr;
    T _arr[N];
};
#endif

} // namespace end of PyTZone  

#endif

