#ifndef __LAYER_H__
#define __LAYER_H__

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <ptz_defs.h>
#include <common_ta.h>
#include <pack.h>

struct layer_TA;
struct network_TA;

typedef struct layer_TA_s layer_TA;

// 这里设计数据结构尽量避免申请堆空间(除了权重数据或是输入输出数据)
// 对于小块的数据在栈上浪费些许内存是可以接受的
struct layer_TA_s {

    struct network_TA *net;

    void (*_forward_ta) (layer_TA *layerta, struct network_TA *net);
    FLOATTA (*_activate_TA) (FLOATTA);
    // void (*) (layer_TA *layerta, struct network_TA *net);
    // void (*) (layer_TA *layerta, struct network_TA *net);
    // ......

    LAYER_TYPE_TA _type;
    DIMENSIONALITY_TA _dimsNum;

    // num c h w  
    INT_TA _dims[MAX_CONV_DIMENSIONS_TA];
    // INT_TA *_dims;
    // num c h w 
    INT_TA _out_dims[MAX_CONV_DIMENSIONS_TA];
    // INT_TA *_out_dims;
                    
    /* FLOATTA *_spatial_mean;   // 空间均值 可能用于批归一化 */
    /* int8_t _adam;              // 是否使用 adam 优化器 */

    int8_t _binary, _xnor;     // 是否使用二进制卷积(binary) 或 XNOR 卷积
    int8_t _keepIn;
    int8_t _keepOut;

    FLOATTA *_mean, *_variance;         // 平均值和方差 用于批归一化

    /* int8_t _batch_normalize; */

    // ACTIVATION_TA _activation;   
    // COST_TYPE_TA _cost_type;
    INT_TA _batch;
    // 从 CA 传递过来的 inputs 默认不计算 batch
    INT_TA _inputs, _outputs;

    // FLOATTA* _weights, _biases;  error
    FLOATTA *_weights, *_biases;        // 权重和偏置项指针 
    INT_TA _nweights, _nbiases;
    // 用于分别存储卷积层的前向传播输出结果和反向传播中的误差(delta)
    // _output: 存储了该卷积层处理输入数据后产生的输出特征图 
    //这个结果将作为下一层的输入 或 作为中间结果 
    // 等价于 _batch * _outputs
    // _delta: 卷积层在反向传播过程中存储的误差项 代表当前层输出与真实目标值之间的误差
    // _delta 记录了每个神经元的梯度 (本项目暂且不考虑权重的更新)
    // FLOATTA *_output, *_delta;              
    FLOATTA *_input; // NULL
    FLOATTA *_output; // NULL
    FLOATTA *_indexes;  // 可用于存储池化过程中的索引位置
    /* FLOATTA _learning_rate_scale; // 学习率的缩放因子 */
    /* FLOATTA *_cost;   // 存储损失值 */
    /* int8_t _random;    // 是否进行随机化 */
    /* int8_t _isloss;    // 是否计算损失 */
    INT_TA _workspace_size; 
    int32_t _idx; 

    // ======================================================
    // ================================================= CONV

    INT_TA _channel;
    INT_TA _num;    // 卷积核的数量

    // 这里的指针的长度通过 _dimsNum 体现
    // CustomArray<int64_t, 2> _size;   // 卷积核大小
    // INT_TA *_size;
    INT_TA _size[MAX_CONV_DIMENSIONS_TA];

    // CustomArray_def<int64_t, 2, 1> _stride;
    // INT_TA *_stride;
    INT_TA _stride[MAX_CONV_DIMENSIONS_TA];

    // CustomArray_def<int64_t, 2, 1> _padding;
    // INT_TA *_padding;
    INT_TA _padding[MAX_CONV_DIMENSIONS_TA];
    
    /* dilation 空洞卷积的扩张率 定义卷积核元素之间的间隔 默认为 1 表示标准卷积 */
    // CustomArray_def<int64_t, 2, 1> _dilation;
    // INT_TA *_dilation;
    INT_TA _dilation[MAX_CONV_DIMENSIONS_TA];

    INT_TA _groups;
    int8_t _isBias;
    /* padding_mode 卷积填充模式 */
    PADDING_MODE_TA _padding_mode;

    // ======================================================
    // =============================================== NormNd
    // size_t _in_c;   Channel _dims[1]
    FLOAT64_TA _eps;
    FLOAT64_TA _momentum;
    int8_t _affine;
    int8_t _track_running_stats;
    // ========================================================
    // ================================================== Activ
    ACTIVATION_TA _activ;
    // ========================================================
    // ================================================= PoolNd 
    // _size _stride _padding _dilation
    int8_t _return_indices; // 默认为 false 
    // 如果为 true 则返回池化时的索引（作用于反向传播时恢复最大值位置）
    int8_t _ceil_mode; // 默认为 false
    int8_t _count_include_pad;
    INT_TA _divisor_override;
    // 如果为 true 池化的输出尺寸向上取整
    // PADDING_MODE _padding_mode;
    // ========================================================
    // ================================================= Linear 
    // int64_t _in_c;   Channel _dims[1]
    // int64_t _out_c;  _out_dims[1]
    // int8_t _bias;    _isBias
    // ========================================================
    // ================================================== tsops
    STATION_TA _stn;
    int64_t _index2;
    int64_t _index1;
    int8_t _kpIdx2;
    int8_t _kpIdx1;

    // ========================================================
    /* device */
    /* dtype 见的类型包括 torch.float32 和 torch.float64 */

};

#if CHECK
void printLayer(layer_TA *layerta);
#endif

INT_TA getKernelSize(const layer_TA * const layerta);
INT_TA getOutSize_c(const layer_TA * const layerta);
INT_TA getInSize_c(const layer_TA * const layertaa);
int dataKeep(layer_TA *l, struct network_TA *net);

#endif