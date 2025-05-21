/*** 
 * @                       _oo0oo_
 * @                      o8888888o
 * @                      88" . "88
 * @                      (| -_- |)
 * @                      0\  =  /0
 * @                    ___/`---'\___
 * @                  .' \\|     |// '.
 * @                 / \\|||  :  |||// \
 * @                / _||||| -:- |||||- \
 * @               |   | \\\  - /// |   |
 * @               | \_|  ''\---/''  |_/ |
 * @               \  .-\__  '-'  ___/-. /
 * @             ___'. .'  /--.--\  `. .'___
 * @          ."" '<  `.___\_<|>_/___.' >' "".
 * @         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
 * @         \  \ `_.   \_ __\ /__ _/   .-` /  /
 * @     =====`-.____`.___ \_____/___.-`___.-'=====
 * @                       `=---='
 * @
 * @
 * @     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * @
 * @           佛祖保佑     永不宕机     永无BUG
 */

#include "./ops/Conv.h"
#include "./ops/Linear.h"
#include "./ops/NormBatch.h"
#include "./ops/Pool.h"
#include "./ops/Activ.h"
#include "./ops/tsops.h"
// #include <core/utils/Serializer.h>
#include "./utils/Serializer.h"

#include <RunMgr.h>

#if LIBTORCH_EXT

using std::array;
using std::vector;
using std::tuple;
// 两者的初始化过程似乎并不兼容
// using c10::ivalue::Tuple;
/* namespace py = pybind11; */


using namespace PyTZone;
using namespace PyTZone::core;

#define PYBIND11 0
#define TORCHSCRIPT 1

#if PYBIND11 

void pybind11_enums(pybind11::module_ &m) {
    // py::enum_<LAYER_TYPE>(m, "LAYER_TYPE")
    //     .value("NULLTYPE", LAYER_TYPE::NULLTYPE)
    //     .value("CONV", LAYER_TYPE::CONV)
    //     .export_values();  // 允许通过 LAYER_TYPE.NULLTYPE 访问
    py::enum_<PADDING_MODE>(m, "PADDING_MODE")
        .value("ZEROS", PADDING_MODE::ZEROS)
        .export_values();  // 允许通过 PADDING_MODE.ZEROS 访问
}

// pybind11 不支持直接绑定未实例化的模板类
// 因此需要为具体的模板实例化进行绑定
void pybind11_CustomArrays(pybind11::module_ &m) {
    py::class_<CustomArray_def<int64_t , 2, 1>>(m, "CustomArray_def<int64_t , 2, 1>")
        .def(py::init<const std::vector<int64_t > &>());
    py::class_<CustomArray_def<int64_t , 2, 0>>(m, "CustomArray_def<int64_t , 2, 0>")
        .def(py::init<const std::vector<int64_t > &>());
    py::class_<CustomArray<int64_t , 2>>(m, "CustomArray<int64_t , 2>")
        .def(py::init<const std::vector<int64_t > &>());
}

// 设置为友元函数 无法解决 pybind11 访问权限受限的问题 
// 本项目也并不建议直接创建 Layer 或 ConvNd 的实例
// 这里直接选择不绑定基类的构造函数
void pybind11_Conv2d(pybind11::module_ &m) {
    // py::nodelete 是 pybind11 中提供的一种策略 表示对象不应被自动销毁 
    // 即当 Python 解释器不再使用对象时 不会调用该对象的析构函数
    py::class_<Layer, std::unique_ptr<Layer, py::nodelete>>(m, "Layer");
        // .def(py::init<>());
    // 绑定 ConvNd 类 继承自 Layer
    py::class_<ConvNd, Layer, std::unique_ptr<ConvNd, py::nodelete>>(m, "ConvNd");
        // .def(py::init<>());
    // 这里涉及到 pybind11 对于继承关系中基类的可见性 所以需要绑定基类 
    py::class_<Conv2d, ConvNd>(m, "Conv2d")
        // 绑定构造函数，可以使用 'py::arg()'为构造函数参数命名 并且可以提供默认值
        // pybind11 使用了 py::init<...>() 模板函数来绑定 C++ 类的构造函数
        .def(py::init<int64_t , int64_t ,
                      const vector<int64_t > &,
                      const vector<int64_t > &,
                      const vector<int64_t > &,
                      const vector<int64_t > &,
                      int64_t , int8_t ,
                      PADDING_MODE>(), 
                      // 这里下面参数要写齐(包括没有默认值的参数)
                      py::arg("channel"),
                      py::arg("num"),
                      py::arg("size"),
                      py::arg("stride") = vector<int64_t >(), 
                      py::arg("padding") = vector<int64_t >(),
                      py::arg("dilation") = vector<int64_t >(),
                      py::arg("groups") = 1,
                      py::arg("isBias") = true,
                      py::arg("padding_mode") = ZEROS)
        // .def(py::init<int64_t , int64_t ,
        //               const vector<int64_t > &,
        //               const tuple<int64_t , int64_t > &,
        //               const tuple<int64_t , int64_t > &,
        //               const tuple<int64_t , int64_t > &,
        //               int64_t , int8_t ,
        //               PADDING_MODE>(), 
        //               // 这里下面参数要写齐(包括没有默认值的参数)
        //               py::arg("channel"),
        //               py::arg("num"),
        //               py::arg("size"),
        //               py::arg("stride") = std::tuple<int64_t , int64_t >(1, 1),
        //               py::arg("padding") = std::tuple<int64_t , int64_t >(1, 1),
        //               py::arg("dilation") = std::tuple<int64_t , int64_t >(1, 1),
        //               py::arg("groups") = 1,
        //               py::arg("isBias") = true,
        //               py::arg("padding_mode") = ZEROS)
        // static_cast: C++ 的强制类型转换操作 用于在存在多个重载时选择特定版本的函数
        // 以下为 const 版本成员函数的设置 as demo
        /* .def("__call__", static_cast<at::Tensor (Conv2d::*)(at::Tensor) const>(&Conv2d::operator())) */
        // .def("__call__", static_cast<at::Tensor (Conv2d::*)(at::Tensor &)>(&Conv2d::operator()))
        // .def("__call__", static_cast<at::Tensor (Conv2d::*)()>(&Conv2d::operator()));  
        .def("__call__", static_cast<void (Conv2d::*)(at::Tensor &)>(&Conv2d::operator()))
        .def("__call__", static_cast<void (Conv2d::*)()>(&Conv2d::operator()));  


        // 对于普通的成员函数
        /* .def("forward", &Conv2d::forward, py::arg("input")) */
        // 绑定只读属性
        /* .def_property_readonly("in_channels", &Conv2d::in_channels) */
        /* .def_property_readonly("out_channels", &Conv2d::out_channels); */
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	// 绑定部分
// TORCH_LIBRARY(my_extension, m) {

    // pybind11_CustomArrays(m);
    pybind11_enums(m);
    pybind11_Conv2d(m);

    // ==================================================================================
    // ==================================================================================
    // ==================================================================================
    /* m.def("relu_ca", &relu_ca_forward, "ReLU_ca forward"); */
    /* m.def("MaxPool2d_ca", &MaxPool2d_ca_forward, "MaxPool2d_ca forward"); */
    /* m.def("Conv2d_ca", &Conv2d_ca_forward, "Conv2d_ca forward"); */
    /* m.def("func", &func, "Conv2d_ca forward"); */

    // m.def("my_function", static_cast<void(*)(size_t, size_t)>(&my_function),
    //     pybind11::arg("a") = 10, pybind11::arg("b") = 20,
    //     "Conv2d_ca forward");

    /* m.def("my_function", static_cast<void(*)(const std::vector<size_t> &)>(&my_function), */
    /*         pybind11::arg("vec") = std::vector<size_t>(), */
    /*         "Conv2d_ca forward"); */

    // m.def("my_function", static_cast<void(*)()>(&my_function), "Conv2d_ca forward");
    // m.def("my_function", &my_function, "Conv2d_ca forward");
    // m.def("Point", &Point::func, "Conv2d_ca forward_from_Point");
    // m.def("Point", []() {
    //     return py::cpp_function(&Point::func);
    // }, "A function that returns a function.");
    
    /* py::class_<Point>(m, "Point") */
    /*     .def(py::init<size_t, size_t, double, std::string>())  // 绑定构造函数 */
    /*     .def("__call__", &Point::operator());                  // 绑定函数调用运算符 */
        //.def("func", &func);  // 绑定成员函数
    // ==================================================================================
    // ==================================================================================
    // ==================================================================================
}

// ====================================================================================================================

// static void TORCH_LIBRARY_init_##ns(torch::Library&);

// c10::Tag是一个enum类型 表示c10::IValue里保存的是什么类型数据 (支持以下类型)
// #define TORCH_FORALL_TAGS(_) \
//   _(None)                    \
//   _(Tensor)                  \
//   _(Storage)                 \
//   _(Double)                  \
//   _(ComplexDouble)           \
//   _(Int)                     \
//   _(SymInt)                  \
//   _(SymFloat)                \
//   _(SymBool)                 \
//   _(Bool)                    \
//   _(Tuple)                   \
//   _(String)                  \
//   _(Blob)                    \
//   _(GenericList)             \
//   _(GenericDict)             \
//   _(Future)                  \
//   _(Await)                   \
//   _(Device)                  \
//   _(Stream)                  \
//   _(Object)                  \
//   _(PyObject)                \
//   _(Uninitialized)           \
//   _(Capsule)                 \
//   _(RRef)                    \
//   _(Quantizer)               \
//   _(Generator)               \
//   _(Enum)

#endif // PYBIND11

#ifdef TORCHSCRIPT 

// 您可能还需要将自定义类移入或移出 IValue``s 例如从 TorchScript 方法中获取或返回“IValue”时 
// 或者您想要在 C++ 中实例化自定义类属性时 要从自定义 C++ 类实例创建一个 ``IValue

// torch::make_custom_class<T>() 提供了一个类似于 c10::intrusive_ptr<T> 的 API 
// 它将获取您提供给它的任何参数集 调用与该参数集匹配的 T 的构造函数 
// 然后将该实例包装起来并返回它 但是 它不是仅返回指向自定义类对象的指针 
// 而是返回包装该对象的 IValue 然后 您可以将此 IValue 直接传递给 TorchScript 

// 如果您已经有一个指向您的类的 intrusive_ptr 则可以使用构造函数 IValue(intrusive_ptr<T>) 直接从中构造一个 IValue 

// 要将 IValue 转换回自定义类
// IValue::toCustomClass<T>() 将返回一个指向 IValue 所包含的自定义类的 intrusive_ptr<T> 
// 在内部 此函数将检查 T 是否已注册为自定义类 以及 IValue 实际上是否包含自定义类 您可以通过调用 isCustomClass() 手动检查 IValue 是否包含自定义类

/* void torch_Serializer(torch::Library &m) { */
/*     m.class_<ConvNd>("ConvNd"); */
/*     m.class_<ConvOpsCtx>("ConvOpsCtx") */
/*         // 我理解不了这里的函数绑定过程 不支持左值引用 和 右值引用 */
/*         // 反正不加 const 就是报错 即使你绑定的函数没有调用 也会在编译时报错 */
/*         // 本来想将这个序列化过程解耦出来 */ 
/*         // 但是这 torchScript 的函数注册过程真的太操蛋了 那干脆不搞了 */
/*         .def(torch::init<std::vector<int64_t>, */
/*                         at::Tensor, */
/*                         std::optional<at::Tensor>, */
/*                         std::vector<int64_t>, */
/*                         std::vector<int64_t>, */
/*                         int8_t, int8_t, */
/*                         int64_t, int64_t, */
/*                         std::vector<int64_t>, */
/*                         std::vector<int64_t>, */
/*                         std::vector<int64_t>, */
/*                         std::vector<int64_t>, */
/*                         int64_t, */
/*                         int8_t, */
/*                         int64_t, int64_t>()) */
/*         .def(torch::init<ConvNd>()) */
/*         .def(torch::init<>()) */
/*         .def_pickle( */
/*             // class intrusive_ptr final { : pytorch/c10/util/intrusive_ptr.h */
/*             [](const c10::intrusive_ptr<ConvOpsCtx>& self) */
/*                 -> SerConvNdPrePack { */
/*                 return (*self.get()).pack(); */
/*             }, */
/*             [](SerConvNdPrePack state) */
/*                 -> c10::intrusive_ptr<ConvOpsCtx> { */
/*                     // return c10::make_intrusive<ConvOpsCtx>( */
/*                     //     std::get<0>(std::get<0>(state)), */
/*                     //     std::get<1>(std::get<0>(state)), */
/*                     //     std::get<2>(std::get<0>(state)), */
/*                     //     std::get<1>(state), */
/*                     //     std::get<2>(state), */
/*                     //     std::get<3>(state), */
/*                     //     std::get<4>(state), */
/*                     //     std::get<5>(state), */
/*                     //     std::get<6>(state), */
/*                     //     std::get<7>(state), */
/*                     //     std::get<8>(state), */
/*                     //     std::get<9>(state), */
/*                     //     std::get<10>(state), */
/*                     //     std::get<11>(state), */
/*                     //     std::get<12>(state), */
/*                     //     std::get<13>(state), */
/*                     //     std::get<14>(state)); */
/*                     return c10::make_intrusive<ConvOpsCtx>(); */
/*             } */
/*         ); */
/* } */

void torch_Ops(torch::Library &m) {

// IValue(c10::intrusive_ptr<ivalue::Tuple> v);

    // m.class_<Conv2d>("Conv2d_ca")
    //     .def(torch::init<int64_t , int64_t ,
    //                   const vector<int64_t > &,
    //                   const tuple<int64_t , int64_t > &,
    //                   const tuple<int64_t , int64_t > &,
    //                   const tuple<int64_t , int64_t > &, 
    //                   int64_t , int8_t ,
    //                   PADDING_MODE>(), 
    //                   // 这里下面参数要写齐(包括没有默认值的参数)
    //                   torch::arg("channel"),
    //                   torch::arg("num"),
    //                   torch::arg("size"),
    //                 //   torch::arg("stride") = c10::IValue(std::tuple<int64_t , int64_t >(1, 1)),
    //                 //   torch::arg("padding") = c10::IValue(std::tuple<int64_t , int64_t >(1, 1)),
    //                 //   torch::arg("dilation") = c10::IValue(std::tuple<int64_t , int64_t >(1, 1)),
    //                   torch::arg("stride"),
    //                   torch::arg("padding"),
    //                   torch::arg("dilation"),
    //                   torch::arg("groups") = 1,
    //                   torch::arg("isBias") = true,
    //                   torch::arg("padding_mode") = ZEROS);

    // void (Conv2d::*attachWithInput)(at::Tensor&) = static_cast<void (Conv2d::*)(at::Tensor&)>( &Conv2d::attach );
    // void (Conv2d::*attachWithoutInput)() = static_cast<void (Conv2d::*)()>(&Conv2d::attach);

    // m.class_<ConvNd>("ConvNd");
    m.class_<Conv2d>("Conv2d")
        // 这里这个操蛋的 TorchScript 函数注册不支持构造函数重载 所以不能注册重载
        // 但是反而可以直接用
        // .def(torch::init<SerConvNdPrePack>()>)
        .def(torch::init<int64_t, int64_t,
                         const vector<int64_t> &,
                         const vector<int64_t> &,
                         const vector<int64_t> &,
                         const vector<int64_t> &,
                         int64_t, int8_t,
                         std::string>())
        // .def(torch::init<>())
        // 这里为什么不用引用 原因参上(ConvOpsCtx 已被注释)
        // 但是 tuple 作为 C++ 11 标准库结构 本身带有移动构造
                    //   这里绑定 torchscript 默认参数设置调试不通
                    //   将以下逻辑进一步封装一层吧
                    //   torch::arg("channel"),
                    //   torch::arg("num"),
                    //   torch::arg("size"),
                    //   torch::arg("stride"),
                    //   torch::arg("padding"),
                    //   torch::arg("dilation"),
                    //   torch::arg("groups") = 1,
                    //   torch::arg("isBias") = true,
                    //   torch::arg("padding_mode") = "ZEROS")
        // 这里 torchScript 目前似乎不支持函数调用运算符绑定 
        // 暂时 找不到绑定的方式 (有兴趣可以去源码中找一下 我是tm懒得找了) 就先用 attach 代替了
        // .def("__call__", static_cast<void (Conv2d::*)(at::Tensor &)>(&Conv2d::operator()))
        // .def("__call__", static_cast<void (Conv2d::*)()>(&Conv2d::operator()))
        
        // 其次也不允许命名空间中函数名重复 
        // 将 Conv2d 类的 attach 成员函数转换为一个函数指针
        // void (Conv2d::*)(/* 参数类型 */) 一个成员函数的函数指针类型
        .def("resetAndAttach", static_cast<void (Conv2d::*)(at::Tensor&)>(&Conv2d::attach))
        .def("attach", static_cast<void (Conv2d::*)()>(&Conv2d::attach))
        .def("Keep", static_cast<void (Conv2d::*)(int8_t, int8_t)>(&Conv2d::setKeep))
        .def("getIdx", static_cast<int64_t (Conv2d::*)() const>(&Conv2d::getIdx))
        .def_pickle(
            // class intrusive_ptr final { : pytorch/c10/util/intrusive_ptr.h
            // 这里的序列化过程待修改 需要结合 TEE 的安全存储接口
            // 这里一般只对考虑权重数据加密 不考虑模型结构的机密性
            [](const c10::intrusive_ptr<Conv2d>& self)
                -> SerConvNdPrePack {
                return (*self.get()).makePack();
            },
            [](SerConvNdPrePack state)
                -> c10::intrusive_ptr<Conv2d> {
                    // 这里直接用移动语义 虽然我也不清楚 state 的生命周期 
                    // 但是 pytorch 源码就是这么用的
                    // std::cout << std::get<1>(std::get<0>(state)) << std::endl;
                return c10::make_intrusive<Conv2d>(std::move(state));
            }
        );
    m.class_<Linear>("Linear")
        .def(torch::init<int64_t, int64_t, int8_t>())
        .def("resetAndAttach", static_cast<void (Linear::*)(at::Tensor&)>(&Linear::attach))
        .def("attach", static_cast<void (Linear::*)()>(&Linear::attach))
        .def("Keep", static_cast<void (Linear::*)(int8_t, int8_t)>(&Linear::setKeep))
        .def("getIdx", static_cast<int64_t (Linear::*)() const>(&Linear::getIdx))
        .def_pickle(
            [](const c10::intrusive_ptr<Linear>& self)
                -> SerLinearPrePack {
                return (*self.get()).makePack();
            },
            [](SerLinearPrePack state)
                -> c10::intrusive_ptr<Linear> {
                return c10::make_intrusive<Linear>(std::move(state));
            }
        );
    m.class_<BatchNorm2d>("BatchNorm2d")
        .def(torch::init<int64_t,
                         double, double,
                         int8_t, int8_t>())
        .def("resetAndAttach", static_cast<void (BatchNorm2d::*)(at::Tensor&)>(&BatchNorm2d::attach))
        .def("attach", static_cast<void (BatchNorm2d::*)()>(&BatchNorm2d::attach))
        .def_pickle(
            [](const c10::intrusive_ptr<BatchNorm2d>& self)
                -> SerNormNdPrePack {
                return (*self.get()).makePack();
            },
            [](SerNormNdPrePack state)
                -> c10::intrusive_ptr<BatchNorm2d> {
                return c10::make_intrusive<BatchNorm2d>(std::move(state));
            }
        );
    m.class_<MaxPool2d>("MaxPool2d")
        .def(torch::init<const vector<int64_t> &,
                         const vector<int64_t> &,
                         const vector<int64_t> &,
                         const vector<int64_t> &,
                         int8_t, int8_t>())
        .def("resetAndAttach", static_cast<void (MaxPool2d::*)(at::Tensor&)>(&MaxPool2d::attach))
        .def("attach", static_cast<void (MaxPool2d::*)()>(&MaxPool2d::attach))
        .def_pickle(
            [](const c10::intrusive_ptr<MaxPool2d>& self)
                -> SerPoolPrePack {
                return (*self.get()).makePack();
            },
            [](SerPoolPrePack state)
                -> c10::intrusive_ptr<MaxPool2d> {
                return c10::make_intrusive<MaxPool2d>(std::move(state));
            }
        );
    m.class_<AvgPool2d>("AvgPool2d")
        .def(torch::init<const vector<int64_t> &,
                         const vector<int64_t> &,
                         const vector<int64_t> &,
                         int8_t, int8_t, int64_t>())
        .def("resetAndAttach", static_cast<void (AvgPool2d::*)(at::Tensor&)>(&AvgPool2d::attach))
        .def("attach", static_cast<void (AvgPool2d::*)()>(&AvgPool2d::attach))
        .def_pickle(
            [](const c10::intrusive_ptr<AvgPool2d>& self)
                -> SerPoolPrePack {
                return (*self.get()).makePack();
            },
            [](SerPoolPrePack state)
                -> c10::intrusive_ptr<AvgPool2d> {
                return c10::make_intrusive<AvgPool2d>(std::move(state));
            }
        );
    // 这里虽然给了接口但是并不建议像 CONV 一样在 python 中声明 Activ
    m.class_<Activ>("Activ")
        .def(torch::init<std::string>())
        .def("resetAndAttach", static_cast<void (Activ::*)(at::Tensor&)>(&Activ::attach))
        .def("attach", static_cast<void (Activ::*)()>(&Activ::attach))
        .def_pickle(
            [](const c10::intrusive_ptr<Activ>& self)
                -> SerActivPrePack {
                return (*self.get()).makePack();
            },
            [](SerActivPrePack state)
                -> c10::intrusive_ptr<Activ> {
                return c10::make_intrusive<Activ>(std::move(state));
            }
        );
    m.class_<TStation>("TStation")
        .def(torch::init<int64_t, int64_t, int8_t, int64_t, int8_t>())
        .def("resetAndAttach", static_cast<void (TStation::*)(at::Tensor&)>(&TStation::attach))
        .def("attach", static_cast<void (TStation::*)()>(&TStation::attach))
        .def("Keep", static_cast<void (TStation::*)(int8_t, int8_t)>(&TStation::setKeep))
        .def("getIdx", static_cast<int64_t (TStation::*)() const>(&TStation::getIdx))
        .def_pickle(
            [](const c10::intrusive_ptr<TStation>& self)
                -> SerTStationPrePack {
                return (*self.get()).makePack();
            },
            [](SerTStationPrePack state)
                -> c10::intrusive_ptr<TStation> {
                return c10::make_intrusive<TStation>(std::move(state));
            }
        );
}

void torch_Net(torch::Library &m) {
    m.def("paramShift", static_cast<void (*)(const std::string &, const at::Tensor &, int64_t)>(paramShift));
    m.def("printNet",  static_cast<void (*)()>(printNet));
    m.def("getTensor",  static_cast<at::Tensor (*)()>(getTensor));
    m.def("setupNet",  static_cast<void (*)()>(setupNet));
    m.def("resetNet",  static_cast<void (*)(at::Tensor &)>(resetNet));
    m.def("partionLayer",  static_cast<void (*)()>(partionLayer));
    m.def("HeapAllocTrack",  static_cast<void (*)()>(HeapAllocTrack));
    m.def("setTorchZone",  static_cast<void (*)(int8_t)>(setTorchZone));
#if TORCHZONE
    m.def("initTee",  static_cast<void (*)()>(initTee));
    m.def("destoryTee",  static_cast<void (*)()>(destoryTee));
#endif
    // m.def("paramShift", paramShift);
}

void torch_utils(torch::Library &m) {
    // at::Tensor makeTensor(FLOATCA *data, const std::vector<int64_t> &sizes); 
    // m.def("makeTensor", static_cast<at::Tensor (*)(FLOATCA *, const std::vector<int64_t> &)>(makeTensor));
}

//   对于模板类不能注册未特化的模板类
//   对于非模板类 可以直接将类名传递作为模板参数
// - 传递给构造函数的参数构成了类的 限定名
//   注册的类在 Python 和 C++ 中将显示为 `torch.classes.my_classes.MyStackClass`。
//   第一个参数称为“命名空间” 第二个参数称为实际的类名

TORCH_LIBRARY(PyTZone, m) {
    torch_utils(m);
    /* torch_Serializer(m); */
    torch_Ops(m);
    torch_Net(m);

//   m.class_<MyStackClass<std::string>>("MyStackClass")
//     // 下一行注册了 MyStackClass 类的构造函数，
//     // 该构造函数接受一个 `std::vector<std::string>` 参数，
//     // 即暴露了 C++ 方法 `MyStackClass(std::vector<T> init)` 
//     // 当前不支持注册重载构造函数 因此现在只能 `def()` 一个实例
//     .def(torch::init<std::vector<std::string>>())
//     // 将一个无状态的(即没有捕获)C++ lambda 函数注册为方法
//     // 注意，lambda 函数必须将 `c10::intrusive_ptr<YourClass>`(或其 const/ref 版本)
//     // 作为第一个参数 其他参数任意
//     .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
//       return self->stack_.back();
//     })
//     // 暴露 MyStackClass<std::string> 类的方法
//     // `torch::class_` 将自动检查传入的方法指针的参数和返回类型
//     // 并将这些信息暴露给 Python 和 TorchScript
//     // 最后 必须获取完整限定方法名的*地址*
//     .def("push", &MyStackClass<std::string>::push)
//     .def("pop", &MyStackClass<std::string>::pop)
//     .def("clone", &MyStackClass<std::string>::clone)
//     .def("merge", &MyStackClass<std::string>::merge)
//   
// // class_<>::def_pickle 允许你为 C++ 类定义序列化和反序列化方法
// // 目前 只支持将无状态的 lambda 函数作为参数传递给 def_pickle
// .def_pickle(
//     // __getstate__
//     // 该函数定义了在序列化该类实例时应该生成什么数据结构
//     //     该函数必须接受一个单一的 `self` 参数，
//     //     这是指向对象实例的 intrusive_ptr
//     //     函数可以返回任何作为 TorchScript 自定义运算符 API 的返回值支持的类型
//     //     在这个例子中，我们选择返回 std::vector<std::string>
//     //     作为要从类中保留的显著数据
//     [](const c10::intrusive_ptr<MyStackClass<std::string>>& self)
//         -> std::vector<std::string> {
//         return self->stack_;
//     },
//     // __setstate__
//     // 该函数定义了在反序列化时如何创建 C++ 类的新实例
//     //     该函数必须接受一个与 `__getstate__` 返回值相同类型的单一参数
//     //     函数必须返回一个指向新实例的 intrusive_ptr
//     //     该实例根据你想要的序列化状态进行初始化
//     [](std::vector<std::string> state)
//         -> c10::intrusive_ptr<MyStackClass<std::string>> {
//         // 一个方便的方法来实例化对象并获取指向它的 intrusive_ptr
//         // 是通过 `make_intrusive`。我们在这里使用它来分配
//         // 一个 MyStackClass<std::string> 的实例，并使用
//         // 序列化状态调用单参数 std::vector<std::string> 构造函数。
//         return c10::make_intrusive<MyStackClass<std::string>>(std::move(state));
//     });
}
#endif // LIBTORCH_EXT
#endif
