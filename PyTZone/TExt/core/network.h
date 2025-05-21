#ifndef __NETWORK_H__
#define __NETWORK_H__

/* typedef struct network_TA{ */
/* 	// 网络的层数 */
/*     int n; */                                            /////////                                
/* 	// 一批训练中的图片参数 和 subdivsions 参数相关 */
/*     int batch; */                                        /////////    
/* 	// 指向网络已处理的图片数量的指针 */
/*     int64_t *seen; */                                     //*******
/*     int *t; */                                           //*******
/* 	// 训练轮数 */
/*     float epoch; */
/* 	// 每个批次分成的小批次数量 */
/*     int subdivisions; */
/*     layer_TA *layers; */                                 //*******
/*     float *output; */
/* 	// 学习率下降策略 */
/*     learning_rate_policy_TA policy; */

/* 	// 当前的学习率 */
/*     float learning_rate; */                              /////////    
/* 	// 动量 用于加速收敛 */
/*     float momentum; */                                   /////////    
/* 	// 权重衰减系数 防止过拟合 */
/*     float decay; */                                      /////////    
/* 	// 与学习率调整相关 */
/*     float gamma; */
/*     float scale; */
/*     float power; */                                      /////////    

/*     int time_steps; */                                   /////////    
/*     int step; */
/*     int max_batches; */                                  /////////    

/* 	// 动态调整学习率 */
/*     float *scales; */
/*     int   *steps; */
/*     int num_steps; */
/*     int burn_in; */                                      /////////    

/* 	// Adam 优化器相关 */
/*     int adam; */                                         /////////    
/*     float B1; */                                         /////////    
/*     float B2; */                                         /////////    
/*     float eps; */                                        /////////    

/* 	// 输入输出维度 */
/*     int inputs; */                                       /////////    
/*     int outputs; */
/*     int truths; */
/*     int notruth; */                                      /////////    
/*     int h, w, c; */                                      /////////    
/*     int max_crop; */                                     /////////    
/*     int min_crop; */                                     /////////    
/*     float max_ratio; */                                  /////////    
/*     float min_ratio; */                                  /////////    
/*     int center; */                                       /////////    
/* 	// 数据增强相关 */
/*     float angle; */                                      /////////    
/*     float aspect; */                                     /////////    
/*     float exposure; */                                   /////////    
/*     float saturation; */                                 /////////    
/*     float hue; */                                        /////////    
/*     int random; */

/*     int gpu_index; */
/*     tree_TA *hierarchy; */

/* 	//中间变量，用来暂存某层网络的输入（包含一个 batch 的输入，比如某层网络完成前向， */
/*     //将其输出赋给该变量，作为下一层的输入，可以参看 network.c 中的forward_network() */
/*     float *input; */
/* 	// 中间变量，与上面的 input 对应，用来暂存 input 数据对应的标签数据（真实数据） */
/*     float *truth; */
/* 	 // 中间变量，用来暂存某层网络的敏感度图（反向传播处理当前层时，用来存储上一层的敏 */
/*     //感度图，因为当前层会计算部分上一层的敏感度图，可以参看 network.c 中的 backward_network() 函数） */
/*     float *delta; */
/* 	// 网络的工作空间, 指的是所有层中占用运算空间最大的那个层的 workspace_size, */
/*     // 因为实际上在 GPU 或 CPU 中某个时刻只有一个层在做前向或反向运算 */
/*     float *workspace; */                                 
/* 	// 网络是否处于训练阶段的标志参数，如果是则值为1. 这个参数一般用于训练与测试阶段有不 */
/*     // 同操作的情况，比如 dropout 层，在训练阶段才需要进行 forward_dropout_layer() */
/*     // 函数， 测试阶段则不需要进入到该函数 */
/*     int train; */
/* 	// 标志参数，当前网络的活跃层 */
/*     int index; */
/* 	//每一层的损失，只有[yolo]层有值 */
/*     float *cost; */                                      //*******
/*     float clip; */                                       /////////    

/*     int64_t workspace_size; */                            //*******

/* } network_TA; */

// make_network_CA(partition_point2 - partition_point1, // n 网络层数
//                 net->learning_rate, 
//                 net->momentum, 
//                 net->decay, 
//                 net->time_steps, 
//                 net->notruth, 
//                 net->batch, 
//                 net->subdivisions, 
//                 net->random, 
//                 net->adam, 
//                 net->B1, 
//                 net->B2, 
//                 net->eps, 
//                 net->h, 
//                 net->w, 
//                 net->c, 
//                 net->inputs, 
//                 net->max_crop, 
//                 net->min_crop, 
//                 net->max_ratio, 
//                 net->min_ratio, 
//                 net->center, 
//                 net->clip, 
//                 net->angle, 
//                 net->aspect, 
//                 net->saturation, 
//                 net->exposure, 
//                 net->hue, 
//                 net->burn_in, 
//                 net->power, 
//                 net->max_batches);

// ========================================================================
// 36
// void make_network_TA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
// {
//     netta.n = n;

//     //netta.seen = calloc(1, sizeof(int64_t));        
//     netta.seen = calloc(1, sizeof(uint64_t));        ************
//     netta.layers = calloc(netta.n, sizeof(layer_TA));************
//     netta.t    = calloc(1, sizeof(int));             ************
//     netta.cost = calloc(1, sizeof(float));           ************

//     netta.learning_rate = learning_rate;
//     netta.momentum = momentum;
//     netta.decay = decay;
//     netta.time_steps = time_steps;
//     netta.notruth = notruth;
//     netta.batch = batch;
//     netta.subdivisions = subdivisions;
//     netta.random = random;
//     netta.adam = adam;
//     netta.B1 = B1;
//     netta.B2 = B2;
//     netta.eps = eps;
//     netta.h = h;
//     netta.w = w;
//     netta.c = c;
//     netta.inputs = inputs;
//     netta.max_crop = max_crop;
//     netta.min_crop = min_crop;
//     netta.max_ratio = max_ratio;
//     netta.min_ratio = min_ratio;
//     netta.center = center;
//     netta.clip = clip;
//     netta.angle = angle;
//     netta.aspect = aspect;
//     netta.saturation = saturation;
//     netta.exposure = exposure;
//     netta.hue = hue;
//     netta.burn_in = burn_in;
//     netta.power = power;
//     netta.max_batches = max_batches;

//     netta.workspace_size = 0;                        ************

//     //netta.truth = net->truth; ////// ing network.c train_network
// }

// ========================================================================

#include "Layer.h"
// #include <RunMgr/RunMgr.h>

namespace PyTZone {
namespace core {

// #define MAX_LAYERS_SEQUENCE 256

class network {
// =========================================================
// =========================================================
public:
    // 这里是为了手动设置参数所写的接口 对于框架本身并不需要或者说是不该存在 
    inline void setNum(int64_t num) {_num = num;}
    inline void setBatch(int64_t batch) {_batch = batch;}
    inline void setInputs(int64_t inputs) {_inputs = inputs;}
    inline void setOutputs(int64_t outputs) {_outputs = outputs;}
    inline void setMemSize(int64_t inoutSize) {_inoutSize = inoutSize;}
    inline void setWorkspaceSize(int64_t workspace_size) {_workspace_size = workspace_size;}
    
    // net 并不管理 input 的生命周期
    inline void setInput(FLOATCA *input) {_input = input;} 
    inline void setOutput(FLOATCA *output) {_output = output;} 
    void printOutput(size_t noutput) const;
// =========================================================
// =========================================================
    // 用于内存计算
    void memUsage_heap() const;
    void initLayerPtrs() {
        for (Layer *l : _Layers) {if (TSTATION_TYPE == l->getType()) l->setLayerPtr();}
    }

public:
    void setLayerThr(LAYER_TYPE type, int32_t idx);
    inline int32_t attachLayer(Layer *pLayer) {
        LogDebug("attachLayer(Layer *)");
        RET_ERROR_CHECK(pLayer, nullptr, " network::attachLayer(Layer *)", INT32_MAX);
        _Layers.push_back(pLayer);
        return _num++;  
    }
    inline int32_t attachLayer(Layer *pLayer, int32_t idx) {
        LogDebug("attachLayer(Layer *, int32_t)");
        RET_ERROR_CHECK(pLayer, nullptr, " network::attachLayer(Layer *)", INT32_MAX);
        auto it = std::lower_bound(_Layers.begin(), _Layers.end(), pLayer, 
                                   [](const Layer *a, const Layer *b){return a->getIdx() < b->getIdx();});
        _Layers.insert(it, pLayer);
        return _num++;  
    }
    // void registerIdx(int32_t idx);
    inline void registerIdx(int32_t idx) {
        _index[++_index[0]] = idx;
    }
    inline void resetLayersIndex() {
        int32_t idx = 0;
        int32_t layerThr_tmp[MAX_LAYERS_SEQUENCE];
        memcpy(layerThr_tmp, _layerThr, MAX_LAYERS_SEQUENCE * sizeof(int32_t)); 

        // std::for_each(_Layers.begin(), _Layers.end(),
        // [&idx, &layerThr_tmp, this](Layer *l) {
        //     int32_t idxOld = l->getIdx();
        //     // cout << "idxOld = " << idxOld << " | idx = " << idx << endl;
        //     _layerThr[idx] = layerThr_tmp[idxOld];
        //     l->setIdx_t(idx++);
        // });
        
        for (auto *l : _Layers) {
            int32_t idxOld = l->getIdx();
            _layerThr[idx] = layerThr_tmp[idxOld];
            l->setIdx_t(idx++);
        }
    
    }
    size_t get_ConvMax_fromThr() const;
    size_t get_ConvMin_fromThr() const;
#if LIBTORCH_EXT
    void partion_Stn();
    // 这里需要重写一个 resetInput 来代替 tensor 的输入
    inline void resetInput(at::Tensor &input) {
        // 这里的 inputs 永远只是浅拷贝
        // if (_input) {
        //     delete[] _input;
        //     _input = nullptr;
        // }
        // todo torch --> float32
        // _input = input.data_ptr<FLOATCA>();
        size_t inputs = input.numel();
        // // =================================================
        // std::cout << "_inputs = " << _inputs << std::endl;
        // printf("address for inputs = %p\n", _input);
        // // =================================================
        memcpy(_input, input.data_ptr<FLOATCA>(), inputs * FLOAT_SIZE);
        _tmp_outputs = inputs;

        auto sizes = input.sizes();
        EXIT_ERROR_CHECK(((sizes.size() + 1) > MAX_CONV_DIMENSIONS), true, "dimensions for inputs error");
        _tmp_dims[0] = sizes.size();
        for (int64_t i = 1; i <= _tmp_dims[0]; ++i)  _tmp_dims[i] = sizes[i - 1];
        // 安全起见 处理剩余的维度
        for (int64_t i = _tmp_dims[0] + 1; i < MAX_CONV_DIMENSIONS; ++i) _tmp_dims[i] = 0; 

        _pre_outputs = inputs;
        _preLayer == nullptr;
    }
#endif // LIBTORCH_EXT
    inline void resetIndex() {
        std::fill(std::begin(_index), std::end(_index), INVALID_VALUE);
        _index[0] = 0;
    }

    inline Layer *getLayer(int32_t idx) {
        RET_ERROR_CHECK(idx + 1 > _Layers.size(), true, "idx in Layers error", nullptr);
        return _Layers[idx];
    }

    inline Layer *getFirstL() {return getLayer(_index[1]);}
    inline Layer *getLastL() {
        int32_t i = _index[0];
        while (0 < i) {
            if (TSTATION_TYPE <= _Layers[_index[i]]->getType()) { --i;continue;}
            else return _Layers[_index[i]];
        }
        return nullptr;
    }
    inline Layer *getPreLayer() {return _preLayer;}
    inline FLOATCA *getOutput() const {return _output;}
    inline int32_t getLayerThr(int32_t idx) const {return _layerThr[idx];}

    inline void updateWorkspaceSize(int64_t workplace_size) {
        (workplace_size > _workspace_size) ? _workspace_size = workplace_size : _workspace_size;
    }
    inline void updateInputs(int64_t inputs) {
        (inputs > _inputs) ? _inputs = inputs : _inputs;
        (_inputs > _inoutSize) ? _inoutSize = _inputs : _inoutSize;
    }
    inline void updateOutputs(int64_t outputs) {
        (outputs > _outputs) ? _outputs = outputs : _outputs;
        (_outputs > _inoutSize) ? _inoutSize = _outputs : _inoutSize;
    }

    inline FLOATCA *getWorkspace() {return _workspace;}
    inline FLOATCA *getInput() {return _input;}
    inline FLOATCA *getOutput() {return _output;}
    inline int64_t getPreOutputs() {return _pre_outputs;}
    inline int64_t getBatch() {return _batch;}
    inline int64_t * const getOutDmis_t() {return _tmp_dims;}

    void makeNet();
    void swapInOutPtr();
    // void resetInOutPtr();
    inline int8_t isTrain() {return _train;}

    void callForward();

#if TORCHZONE
    int callForward_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr);
    void fetchOutput_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr);

    int Forward_Fetch_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr);

    void makeNet_CA(TEEC_INVITATION_T *TEEC_INVITATION = nullptr);
    void makeNet_Pthreads_CA();
#endif

public:
    network();
    ~network();

public:
    void print() const;
private:
	// net.input
	// 网络的工作空间, 指的是所有层中占用运算空间最大的那个层的 workspace_size, 
    // 因为实际上在 GPU 或 CPU 中某个时刻只有一个层在做前向或反向运算
	// net.workspace

    int64_t _num;                // Layer 层数
    int64_t _subNum;
    int64_t _batch;

    // 当 _inoutSize 被初始化完成后 
    // _inputs _outputs 就被用来记录输入输出长度
    int64_t _inputs, _outputs;   // 输入数量和输出数量的最大值
    int64_t _inoutSize;
    // int64_t _high, _weight, _channel;
    int64_t _workspace_size;     

    // 这里的 inputs 目前永远是浅拷贝 所以并不关心其数据的生命周期
    FLOATCA *_input;             
    // 所有作为返回值的 Tensor 都不会做数据的拷贝
    // 其数据生命周期的管理都交给 net
    FLOATCA *_output;

    FLOATCA *_mem1;
    FLOATCA *_mem2;   

    // 理论上这个 _workspace 在 CA 侧用不到
    FLOATCA *_workspace;          // 网络的工作空间 用于存储临时变量以进行计算

    // 剪裁: 是用来限制网络输出范围的参数 通常用于防止输出值过大或过小
    double _clip;    
    int8_t _train;

    // ===============================================================
    // 链表类模板 to attach Layer 
    // 这里在设计上有个问题 : py 脚本 init 函数中 算子注册的顺序 不等于 forward 函数中算子的调用顺序
    // 这导致执行时链表的查找开销
    // 但是 构造函数中无法感知 forward 内的情况

    // 原则上 我们希望 ca 侧的上下文能够一次性初始化完成
    // 并且 ca 侧 Layer 在链表中的顺序 等于 Layer 的执行顺序

    // 考虑到 ca 侧通常不考虑运行开销 并且 初始化的开销通常不考虑
    // ca 侧用 std::vector 初始化为数组
    // ta 侧也用数组形式保存 Layer 结构 通过偏移量参数来保证顺序执行

    std::vector<Layer *> _Layers;
    // num : idx1 idx2 idx3 ......
    int32_t _index[MAX_LAYERS_SEQUENCE];  // 标记当前活跃的层 用于管理网络的推进

    // 在多线程的环境下需要用 net 来进行统一管理
    // Layer 与 Layer 之间相互不可见因此 Layer 对于 threadSize 是否可知影响不大 
    // 这里的下标 0 - n 代表对应的 layer 的 idx 
    // layerThr[idx] == 0 1 2 分别代表 : 单线程执行 多线程无依赖 多线程全依赖
    // 初始化中全部为 0
    int32_t _layerThr[MAX_LAYERS_SEQUENCE]; // 这里初始化 -1

    // ===============================================================
    // 这里主要用来辅助推理
    int64_t _pre_outputs;
    Layer *_preLayer;

    int64_t _tmp_outputs;
    int64_t _tmp_dims[MAX_CONV_DIMENSIONS];
};


extern int8_t Flag_TorchZone;
void setTorchZone(int8_t flag);

#define NETWORK (Singleton<network>::getInstance())

#if LIBTORCH_EXT
void paramShift(const std::string &str, const at::Tensor &data, int64_t idx);
// 需要重写
at::Tensor getTensor();
void resetNet(at::Tensor &input);

// 这里 Layer 在声明多线程调用时需要提供 idx 并将声明信息暂存在 network 中
// 因为目前的 partionLayer 还不支持更新 subNet 部分的 layerIdx 
// 因此需要在 partionLayer 操作之后再进行 subNet 的初始化
// 但是至少 partionLayer 中会更新 layerThr[]
void partionLayer();
#endif // LIBTORCH_EXT

void printNet();

void HeapAllocTrack();

} // namespace end of core
} // namespace end of PyTZone 

#endif

