#ifndef __LAYER_H__
#define __LAYER_H__

#include <RunMgr/all.h>
#include <core/utils/common.h>
#include <core/utils/tensor.h>
#include <core/utils/SerPrePack.h>

/* LAYER_TYPE type: 层的类型(如卷积、全连接、池化等) */
/* ACTIVATION_TA activation: 激活函数的类型 */
/* COST_TYPE_TA cost_type: 损失函数的类型 */
/* void (*forward_TA)(struct layer_TA, struct network_TA): 前向传播函数指针，指向具体的前向传播实现 */
/* void (*backward_TA)(struct layer_TA, struct network_TA): 反向传播函数指针 */
/* void (*update_TA)(struct layer_TA, update_args_TA): 用于更新参数的函数指针 */
/* int batch: 批次大小，表示一次处理的数据样本数量 */
/* int inputs, outputs: 输入、输出的大小或数量 */
/* **float weights, biases: 权重和偏置项指针 */
/* **float output, delta: 输出和反向传播中的误差 */
/* **float mean, variance: 平均值和方差，用于批归一化 */
/* int batch_normalize: 是否进行批归一化 */
/* float learning_rate_scale: 学习率的缩放因子 */
/* *float cost: 存储损失值 */
/* int random: 是否进行随机化操作，通常用于数据增强等 */
/*  workspace_size: 层在计算过程中需要的工作空间大小 */
/* int noloss: 是否不计算损失 */

// 以下是特定操作符相关成员
/* int binary, xnor: 是否使用二进制卷积(binary)、XNOR 卷积 */
/* int adam: 是否使用 Adam 优化器 */
/* float clip: 梯度裁剪，用于防止梯度爆炸 */
/* float jitter: 用于数据增强的抖动值 */
/* **float binary_input, binary_weights: 二进制卷积的输入和权重 */
/* *tree_TA softmax_tree: 用于 Softmax 操作的层次结构 */
/* int softmax: 是否使用 Softmax 激活函数 */
/* int classes: 分类任务中的类别数量 */

/* 卷积层相关成员: */ 
/* int h, w, c: 输入的高度、宽度和通道数 */
/* int out_h, out_w, out_c: 输出的高度、宽度和通道数 */
/* int n: 卷积核的数量（输出通道数） */
/* int size: 卷积核的大小（例如 3x3 卷积） */
/* int stride, padding: 卷积的步幅和填充 */
/* *float spatial_mean: 空间均值，可能用于批归一化 */
/* int groups: 分组卷积的组数 */
/* RNN、LSTM、GRU 层相关成员： */
/* **float state, prev_state: 当前状态和前一状态(RNN、LSTM 等) */
/* **float *z_cpu, r_cpu, h_cpu: GRU 的 z 门、r 门和 h 门相关状态 */
/* **float *f_cpu, *i_cpu, g_cpu, o_cpu: LSTM 的遗忘门、输入门、记忆单元、输出门 */
/* **float c_cpu, dc_cpu: LSTM 的单元状态和状态增量 */
/* **struct layer_TA input_gate_layer, state_gate_layer: 门控单元层，通常与 RNN 或 LSTM 相关 */
/* **struct layer_TA *wz, *uz, *wr, *ur, wh, uh: GRU 层的权重和偏置项 */
// =================================================================================

namespace PyTZone {

struct subNet_s;

namespace core {

class network;


PADDING_MODE stringToPM(const string &str);
bool isParams(LAYER_TYPE type);

class Layer {
// =========================================================
public:
    // 这里是为了手动设置参数所写的接口 对于框架本身并不需要或者说是不该存在 
    inline void setType_t(LAYER_TYPE type) {_type = type;}
    inline void setDimsNum_t(DIMENSIONALITY dimsNum) {_dimsNum = dimsNum;}
    inline void setBinary_t(int8_t binary) {_binary = binary;}
    inline void setXnor_t(int8_t xnor) {_xnor = xnor;}
    inline void setBatch_t(int64_t batch) {_batch = batch;}
    inline void setInputs_t(int64_t inputs) {_inputs = inputs;}
    inline void setOutputs_t(int64_t outputs) {_outputs = outputs;}
    inline void setNweights_t(int64_t nweights) {_nweights = nweights;}
    inline void setNbiases_t(int64_t nbiases) {_nbiases = nbiases;}
    inline void setWorkspaceSize_t(int64_t workspace_size) {_workspace_size = workspace_size;}
    inline void setIdx_t(int32_t idx) {_idx = idx;}
    void setData_randn_t();
    // int64_t _dims[MAX_CONV_DIMENSIONS];
    // int64_t _out_dims[MAX_CONV_DIMENSIONS];
    // void setDims(int64_t dims[MAX_CONV_DIMENSIONS]);
    // void setOutDims(int64_t out_dims[MAX_CONV_DIMENSIONS]);
// =========================================================
    inline const int64_t *getDmis() const {return _dims;}
    inline const int64_t *getOutDmis() const {return _out_dims;}
    inline int64_t * const getDmis_t() {return _dims;}
    inline int64_t * const getOutDmis_t() {return _out_dims;}
    inline int64_t getBatch() const {return _batch;}
    void dataKeep(int64_t inputs, int64_t outputs, int64_t batch);

public:
// 暴露 virtual 接口给 network 
#if LIBTORCH_EXT
    // paramLoad 目前已被弃用
    virtual void paramLoad(const std::string &str, const at::Tensor &data) = 0;
    virtual void initialize(const at::Tensor &data) = 0;
    void setWeights(const at::Tensor &data);
    void setBiases(const at::Tensor &data);
    void setMeans(const at::Tensor &data);
    void setVariances(const at::Tensor &data);
    void setBatch(const at::Tensor &data);
#endif // LIBTORCH_EXT
    virtual void print() const = 0;
    virtual void forward() = 0;
    // virtual void activate() const = 0;

    virtual void setLayerPtr() {return;}
    virtual void memUsage_heap(int64_t &heapAll, int64_t &heapApply, int64_t &heapWeightsOnly) const;
    virtual void updateNet_thr(struct subNet_s &subNet);    // for mainNet
    virtual void updateNet_thr(List<struct subNet_s> &subNet, size_t threadSize) {return;}
    
    inline  int64_t getIdx() const {return _idx;}
    // 单纯为了测试序列化模块而设计的测试函数
    inline void setIdx(int32_t idx) {_idx = idx;}
// 并不建议创建 Layer 实例

    void updateNet();
    inline int64_t getInputs() const {return _inputs;}
    inline int64_t getOutputs() const {return _outputs;}
    inline FLOATCA *getInput() {return _input;}
    inline FLOATCA *getOutput() {return _output;}
    inline LAYER_TYPE getType() const {return _type;}
    inline int8_t getKeepIn() const {return _keepIn;}
    inline int8_t getKeepOut() const {return _keepOut;}

    void registerPthr() const;

    inline int64_t getOutSize_c() const {
        int64_t size = 1;
        for (size_t i = 2; i < _out_dims[0] + 1; ++i) 
            size *= _out_dims[i];
        return size;
    }

    inline int64_t getInSize_c() const {
        int64_t size = 1;
        for (size_t i = 2; i < _dims[0] + 1; ++i) 
            size *= _dims[i];
        return size;
    }


public:
    // ================================================= for ConvNd
    virtual int64_t getNum_ConvNd() const {return 0;}

#if TORCHZONE
    virtual void make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr) const = 0;
    void make_layer_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr) const;
#endif

protected:
// public:
    // 这里的构造函数目前还是缺省的 
    Layer() {};
    // Layer() = delete;
    Layer(LAYER_TYPE type, DIMENSIONALITY dimsNum = DN);
#if SERIALIZER
    Layer(ConvOpsCtx &coc);
#endif
#if LIBTORCH_EXT
    Layer(SerPrePack_Layer &&sppL);
    const SerPrePack_Layer &makeLPack() const;
    void get_tensor_dimensions(const at::Tensor &data, int64_t dims[MAX_CONV_DIMENSIONS]);
    void paramLoad_for_datas(const std::string &str, const at::Tensor &data);
#endif // LIBTORCH_EXT
    // virtual bool checkStr(const std::string &str) const = 0;
    virtual ~Layer();
    void destory();
    // friend void pybind11_Conv2d(pybind11::module_ &);
    std::vector<int64_t> &makeParams() const;
#if SERIALIZER
    const at::Tensor &getWeights() const;
    const at::Tensor &getBiases() const;
#endif  // SERIALIZER
public: // 暴露给 c++ 中用于 qemu 模拟推理
    // 默认对于二维卷积来说处理三维输入
    void setDims(int64_t target[MAX_CONV_DIMENSIONS], 
                 int64_t source[MAX_CONV_DIMENSIONS],
                 int64_t num);
    void setOutDims(int64_t dims[MAX_CONV_DIMENSIONS],
                    int64_t out_dims[MAX_CONV_DIMENSIONS],
                    int64_t num,
                    vector<int64_t> size,
                    vector<int64_t> stride,
                    vector<int64_t> padding);
protected:
    void setDimsV(const std::vector<int64_t> &dims, 
                  int64_t target_dims[MAX_CONV_DIMENSIONS],
                  int64_t num);
    void setOutputs(int64_t out_dims[MAX_CONV_DIMENSIONS]);
    void setWorkspaceSize(int64_t dims[MAX_CONV_DIMENSIONS],
                          int64_t out_dims[MAX_CONV_DIMENSIONS],
                          int64_t groups,
                          vector<int64_t> size,
                          DIMENSIONALITY dimsNum = D2);
    inline int64_t getChannel(int64_t dims[MAX_CONV_DIMENSIONS]) const {
        return dims[1];
    }
    inline int64_t getHigh(int64_t dims[MAX_CONV_DIMENSIONS]) const {
        return dims[2];
    }
    inline int64_t getWeight(int64_t dims[MAX_CONV_DIMENSIONS]) const {
        return dims[3];
    }
    // inline void setpLayers(int64_t index1, int64_t index2) {
    //     if (index1 >= 0) _layer1 = NETWORK->getLayer(index1);
    //     if (index1 >= 0) _layer2 = NETWORK->getLayer(index2);
    // }
protected:

    // void (*_forward) ( *layer, struct network *net);
    FLOATCA (*_activate) (FLOATCA);

    LAYER_TYPE _type;
    DIMENSIONALITY _dimsNum;
    // ACTIVATION_TA _activation;   
    // COST_TYPE_TA _cost_type;
    
    // 其实数组的设计完全是累赘
    // 之所以这么设计是想要将 ConvNd 类的数据成员统一化
    // 对于后续高维的卷积不需要再改变数据成员
    // int64_t _high, _weight;
    // int64_t out_h, out_w;
    // num c h w  
    int64_t _dims[MAX_CONV_DIMENSIONS];
    // num c h w 
    int64_t _out_dims[MAX_CONV_DIMENSIONS];
                
    /* int8_t _adam;              // 是否使用 adam 优化器 */
    int8_t _binary, _xnor;     // 是否使用二进制卷积(binary) 或 XNOR 卷积
    
    int8_t _keepIn;  
    int8_t _keepOut; 
    
// ================================================================
    /* FLOATCA *_spatial_mean;   // 空间均值 可能用于批归一化 */
    // FLOATCA *_scales;   // 存储用于归一化的缩放系数
    FLOATCA *_mean;     
    FLOATCA *_variance;

    /* int8_t _batch_normalize; */
// ================================================================
    int64_t _batch;

    // 通过 (data.numel()) 得到
    int64_t _inputs, _outputs;

    // FLOATCA* _weights, _biases;  error
    FLOATCA *_weights, *_biases;        // 权重(缩放因子)和偏置项指针 
    // at::Tensor _weights, _biases;
    int64_t _nweights, _nbiases;
    // 用于分别存储卷积层的前向传播输出结果和反向传播中的误差(delta)
    // _output: 存储了该卷积层处理输入数据后产生的输出特征图 
    //这个结果将作为下一层的输入 或 作为中间结果 
    // 等价于 _batch * _outputs
    // _delta: 卷积层在反向传播过程中存储的误差项 代表当前层输出与真实目标值之间的误差
    // _delta 记录了每个神经元的梯度 (本项目暂且不考虑权重的更新)
    // FLOATCA *_output, *_delta;              
    
    /* FLOATCA _learning_rate_scale; // 学习率的缩放因子 */
    /* FLOATCA *_cost;   // 存储损失值 */
    /* int8_t _random;    // 是否进行随机化 */
    /* int8_t _isloss;    // 是否计算损失 */
    int64_t _workspace_size; 
    // idx 是唯一一个在序列化过程中需要单独进行数据类型转换的
    int32_t _idx;
    
    // ============================================
    Layer *_layer1, *_layer2;
    FLOATCA *_input, *_output;
    // ============================================
    // int8_t isThreads;   // threads 的标记由 net 负责
};



} // namespace end of core
} // namespace end of PyTZone 

#endif
