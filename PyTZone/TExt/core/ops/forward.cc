#include "Conv.h"
#include "Linear.h"
#include "NormBatch.h"
#include "Pool.h"
#include "Activ.h"
#include "tsops.h"
#include <RunMgr.h>
#include <stdio.h>


namespace PyTZone { 
namespace core {

// extern "C" 只能用于 全局/命名空间作用域
#ifdef __cplusplus
extern "C" {  
#endif
#include <im2col.h>
#include <gemm.h>
#include <math.h>
#ifdef __cplusplus
}
#endif

// 目前 Conv2d、BatchNorm2d 计算有误

void Conv2d::forward() {
    LogDebug("void Conv2d::forward() : idx = %d : ", _idx);
    FLOATCA *NET_INPUT = NETWORK->getInput();
    FLOATCA *NET_OUTPUT = NETWORK->getOutput();
    FLOATCA *NET_WORKSPACE = NETWORK->getWorkspace();

    int channels = this->_channel / this->_groups;
    int m = this->_num / this->_groups;
    int k = getKernelSize() * channels;
    int n = getOutSize_c();

    for(int i = 0; i < this->_batch; ++i){
        for(int j = 0; j < this->_groups; ++j){
            // weight *
            FLOATCA *a = this->_weights + (j * this->_nweights / this->_groups);
            // workplace *
            FLOATCA *data_col = NET_WORKSPACE;
            // 当前分组的输出
            // FLOATCA *c = this->_output + (i * uint32ToInt(this->_groups) + j) * n * m;
            FLOATCA *c = NET_OUTPUT + (i * this->_groups + j) * n * m;
            // 当前的输入特征图
            FLOATCA *data_im =  NET_INPUT + (i * this->_groups + j) * channels * getInSize_c();
            if (this->_size[0] == 1 && this->_size[1] == 1) {
                data_col = data_im;
            } else {
                im2col_cpu_2d(data_im, 
                              channels, this->_dims[2], this->_dims[3],
                              this->_size[0], this->_size[1],
                              this->_stride[0], this->_stride[1],
                              this->_padding[0], this->_padding[1],
                              data_col);
            }
            // ...... todo dimsNum
            // 输出保存在 FLOATCA *c 中
            gemm_cpu(0,0,m,n,k,1,a,k,data_col,n,1,c,n); // nn
        }
    }
    if (this->_isBias) add_bias(NET_OUTPUT, this->_biases, this->_batch, this->_num, n);
    dataKeep(_inputs, _outputs, _batch); 
}

void Linear::forward() {
    LogDebug("void Linear::forward() : idx = %d : ", _idx);
    FLOATCA *NET_INPUT = NETWORK->getInput();
    FLOATCA *NET_OUTPUT = NETWORK->getOutput();
    FLOATCA *NET_WORKSPACE = NETWORK->getWorkspace();

    int m = this->_batch;
    int k = this->_inputs;
    int n = this->_outputs;
    float *a = NET_INPUT;
    float *b = this->_weights;
    float *c = NET_OUTPUT;
    gemm_cpu(0,1,m,n,k,1,a,k,b,k,1,c,n);    // nt
    if (this->_isBias) add_bias(NET_OUTPUT, this->_biases, this->_batch, this->_outputs, 1);
    dataKeep(_inputs, _outputs, _batch);
}

void BatchNorm2d::forward() {
    LogDebug("void BatchNorm2d::forward() : idx = %d : ", _idx);
    FLOATCA *NET_INPUT = NETWORK->getInput();
    FLOATCA *NET_OUTPUT = NETWORK->getOutput();
    FLOATCA *NET_WORKSPACE = NETWORK->getWorkspace();

    int wh = getOutSize_c();
    if(this->_type == BATCHNORM_TYPE) NETWORK->swapInOutPtr();
    if(NETWORK->isTrain()){
        // mean_cpu(l->output, l->batch, l->out_c, l->out_h*l->out_w, l->mean);
        // variance_cpu(l->output, l->mean, l->batch, l->out_c, l->out_h*l->out_w, l->variance);

        // scal_cpu(l->out_c, .99, l->rolling_mean, 1);
        // axpy_cpu(l->out_c, .01, l->mean, 1, l->rolling_mean, 1);
        // scal_cpu(l->out_c, .99, l->rolling_variance, 1);
        // axpy_cpu(l->out_c, .01, l->variance, 1, l->rolling_variance, 1);

        // normalize_cpu(l->output, l->mean, l->variance, l->batch, l->out_c, l->out_h*l->out_w);
        // copy_cpu(l->outputs*l->batch, l->output, 1, l->x_norm, 1);
    } else {
        normalize_cpu(NET_OUTPUT, this->_mean, this->_variance, this->_batch, this->_out_dims[1], wh);
    }
    scale_bias(NET_OUTPUT, this->_weights, this->_batch, this->_out_dims[1], wh);
    add_bias(NET_OUTPUT, this->_biases, this->_batch, this->_out_dims[1], wh);
}

void AvgPool2d::forward() {
    LogDebug("void AvgPool2d::forward() : idx = %d : ", _idx);
    FLOATCA *NET_INPUT = NETWORK->getInput();
    FLOATCA *NET_OUTPUT = NETWORK->getOutput();
    FLOATCA *NET_WORKSPACE = NETWORK->getWorkspace();

    int b,i,k;
    int c = this->_dims[1];

    for(b = 0; b < this->_batch; ++b){
        for(k = 0; k < c; ++k){
            int out_index = k + b*c;
            NET_OUTPUT[out_index] = 0;
            for(i = 0; i < this->_dims[2]*this->_dims[3]; ++i){
                int in_index = i + this->_dims[2]*this->_dims[3]*(k + b*c);
                NET_OUTPUT[out_index] += NET_INPUT[in_index];
            }
            NET_OUTPUT[out_index] /= this->_dims[2]*this->_dims[3];
        }
    }
}

void MaxPool2d::forward() {
    LogDebug("void MaxPool2d::forward() : idx = %d : ", _idx);
    FLOATCA *NET_INPUT = NETWORK->getInput();
    FLOATCA *NET_OUTPUT = NETWORK->getOutput();
    FLOATCA *NET_WORKSPACE = NETWORK->getWorkspace();

    int b,i,j,k,m,n;
    int h_offset = -this->_padding[0];
    int w_offset = -this->_padding[1];

    int h_out = this->_out_dims[2];
    int w_out = this->_out_dims[3];
    int c = this->_dims[1];

    for(b = 0; b < this->_batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h_out; ++i){
                for(j = 0; j < w_out; ++j){
                    int out_index = j + w_out*(i + h_out*(k + c*b));
                    float max = -first_aim_money;
                    // int max_i = -1;
                    for(n = 0; n < this->_size[0]; ++n){
                        for(m = 0; m < this->_size[1]; ++m){
                            int cur_h = h_offset + i*this->_stride[0] + n;
                            int cur_w = w_offset + j*this->_stride[1] + m;
                            int index = cur_w + this->_dims[3]*(cur_h + this->_dims[2]*(k + b*this->_dims[1]));
                            int valid = (cur_h >= 0 && cur_h < this->_dims[2] &&
                                         cur_w >= 0 && cur_w < this->_dims[3]);
                            float val = (valid != 0) ? NET_INPUT[index] : -first_aim_money;
                            // max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    NET_OUTPUT[out_index] = max;
                    // this->indexes[out_index] = max_i;
                }
            }
        }
    }
}

void Activ::forward() {
    LogDebug("void Activ::forward() : idx = %d : ", _idx);
    FLOATCA *NET_INPUT = NETWORK->getInput();
    FLOATCA *NET_OUTPUT = NETWORK->getOutput();
    // FLOATCA *NET_WORKSPACE = NETWORK->getWorkspace();

    uint32_t num = this->_outputs * this->_batch;
    int i;
    for (i = 0; i < num; ++i) {
        NET_OUTPUT[i] = this->_activate(NET_INPUT[i]);
    }
}

void TStation::forward() {
    LogDebug("void TStation::forward() : idx = %d : ", _idx);

    int64_t outputs = NETWORK->getPreOutputs();
    FLOATCA *NET_INPUT = NETWORK->getInput();
    FLOATCA *NET_OUTPUT = NETWORK->getOutput();

    network *net = NETWORK;
    // Layer *l = net->getLayer(0);

    float *dt2 = NULL;
    if (_layer2) {dt2 = (_kpIdx2) ? _layer2->getInput() : _layer2->getOutput();} 
    else {dt2 = NET_INPUT;}

    float *dt1 = NULL;
    if (_layer1) {dt1 = (_kpIdx1) ? _layer1->getInput() : _layer1->getOutput();} 
    else {dt1 = NET_INPUT;}

    if (ADD_STATION == _stn) {
        // outputs = (_kpIdx2) ? _layer2->getInputs() : _layer2->getOutputs();
        for(int64_t i = 0; i < outputs; ++i)
            NET_OUTPUT[i] = dt1[i] + dt2[i];
    } else if (NONE_STATION == _stn) {
        // 理论上即使什么都不做也最好把 input 拷贝给 output
        memcpy(NET_OUTPUT, NET_INPUT, outputs * FLOAT_SIZE);
    }

    if (_keepIn) {_input = new FLOATCA[outputs];}
    if (_keepOut) {_output = new FLOATCA[outputs];}
    dataKeep(outputs, outputs, NETWORK->getBatch());
}

void Activ::setActivate() {
    switch(_activ){
        case LOGISTIC_ACTIV:
            _activate = logistic_activate;break;
        case RELU_ACTIV:
            _activate = relu_activate;break;
        case RELIE_ACTIV:
            _activate = relie_activate;break;
        case LINEAR_ACTIV:
            _activate = linear_activate;break;
        case RAMP_ACTIV:
            _activate = ramp_activate;break;
        case TANH_ACTIV:
            _activate = tanh_activate;break;
        case PLSE_ACTIV:
            _activate = plse_activate;break;
        case LEAKY_ACTIV:
            _activate = leaky_activate;break;
        case ELU_ACTIV:
            _activate = elu_activate;break;
        case LOGGY_ACTIV:
            _activate = loggy_activate;break;
        case STAIR_ACTIV:
            _activate = stair_activate;break;
        case HARDTAN_ACTIV:
            _activate = hardtan_activate;break;
        case LHTAN_ACTIV:
            _activate = lhtan_activate;break;
        case SELU_ACTIV:
            _activate = selu_activate;break;
    }
    return;
}



} // namespace end of core
} // namespace end of PyTZone
