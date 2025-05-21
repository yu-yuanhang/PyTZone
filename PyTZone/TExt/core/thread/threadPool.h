#ifndef __THREADPOOL_H__
#define __THREADPOOL_H__

#include <network.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#include <templateList.h>

namespace PyTZone {

using core::network;
using core::Layer;

#define MAX_QUEUE_SIZE 64

typedef struct subLayer_s
{
    subLayer_s(): 
        inputs(INVALID_VALUE_U), output(INVALID_VALUE_U),
        offsetWeight(INVALID_VALUE_U),
        weightSize(INVALID_VALUE_U),
        offsetBiases(INVALID_VALUE_U),
        biaseSize(INVALID_VALUE_U),
        workspace_size(INVALID_VALUE_U),
        num(INVALID_VALUE_U),
        idx(INVALID_VALUE) {}
    friend std::ostream &operator<<(std::ostream &os, const subLayer_s &subLayer) {
        os << "idx = " << subLayer.idx << endl \
           << "inputs = " << subLayer.inputs << "| output = " << subLayer.output << endl \
           << "offsetWeight = " << subLayer.offsetWeight << endl \
           << "weightSize = " << subLayer.weightSize << endl \
           << "offsetBiases = " << subLayer.offsetBiases << endl \
           << "biaseSize = " << subLayer.biaseSize << endl \
           << "workspace_size = " << subLayer.workspace_size << endl \
           << "num = " << subLayer.num << endl;
        return os;
    }

    int64_t inputs, output;
    // int64_t offsetIn, offsetOut;

    int64_t offsetWeight;
    int64_t weightSize;
    int64_t offsetBiases;
    int64_t biaseSize;

    int64_t workspace_size;

// 除了 idx 以下的成员仅在多核调用中的 subNet 中起作用
// ========================================== CONV 
// 这里 Conv Linear BatchNorm 需要重写
    int64_t num;

// ==========================================
    // 对于 subLayer 在使用 keep 来保存中间结果时 
    // 需要额外关注连续的几层是否在同一次 TA 调用中
    int32_t idx;    // 这里的 idx 区分于 NETWORK 中的 idx 用来在 main/sub Net 标记 subLayer

} subLayer_t;


typedef struct subNet_s {
    subNet_s(): 
        num(INVALID_VALUE_U),
        inputs(INVALID_VALUE_U), output(INVALID_VALUE_U),
        inoutSize(INVALID_VALUE_U),
        workspace_size(INVALID_VALUE_U),
        subLayerArr() {
        std::fill(std::begin(index), std::end(index), INVALID_VALUE);
    }
    friend std::ostream &operator<<(std::ostream &os, const subNet_s &subNet) {
        os << "num = " << subNet.num << endl \
           << "inputs = " << subNet.inputs << "| output = " << subNet.output << endl \
           << "inoutSize = " << subNet.inoutSize << endl \
           << "workspace_size = " << subNet.workspace_size << endl;

        os << "index :";
        for (int i = 0; subNet.index[i] != -1; ++i) os << " " << subNet.index[i];
        os << endl; 
        os << "list for subLayerArr here" << endl;
        subNet.subLayerArr.print();
        return os;
    }
    // void updateNet(Layer *l) {
    //     num++;
    //     int64_t tmp = l->getInputs();
    //     inputs = (tmp > inputs ? tmp : inputs);
    //     tmp = l->getOutputs();
    //     output = (tmp > output ? tmp : output);
    //     inoutSize = (inputs > output ? inputs : output)
    //     // ......
    // }

    int64_t num;    // number of subLayer
    int64_t inputs, output;
    int64_t inoutSize;
    int64_t workspace_size; 

    // 每个 subNet 对应的 Layer 单独保存 
    List<subLayer_t> subLayerArr;
    // 多核多 TA 中的 idx 并不等价与位置下标
    int32_t index[MAX_LAYERS_SEQUENCE];
} subNet_t;



typedef struct threadPool_s 
{
    threadPool_s(): 
        threadSize(INVALID_VALUE_U), tidArr(nullptr),
        subNetArr(), mainNet(), exitflag(false), curSize(INVALID_VALUE_U) {
        std::fill(std::begin(index_Net), std::end(index_Net), INVALID_VALUE);
        pthread_rwlock_init(&rwlock, nullptr);
        pthread_mutex_init(&mtx_index, NULL);
        pthread_mutex_init(&mtx_curSize, NULL);
        pthread_cond_init(&cv, NULL);
    }
    // ~threadPool_s() {}

    void print() const {
        cout << "!!!!!!!!!!!!!! print() :  threadPool !!!!!!!!!!!!!!" << endl \
             << "threadSize         : " << threadSize << endl \
             << "exitflag           : " << (exitflag ? "true" : "false") << endl;
        cout << "mainNet here : " << endl << mainNet << endl;
        cout << "subNetArr here : " << endl;
        subNetArr.print();
    }

    // 在设计上子线程的数量最大为 各卷积层卷积核数的最大值
    size_t threadSize;
    unique_ptr<pthread_t[]> tidArr;     // 子线程数组
    // unique_ptr<int[]> threadIds;        

    // std::vector<Layer *> Layers;

    // 这里应该与子线程一一对应
    List<subNet_t> subNetArr;   // 处理用于多核执行的部分    
    subNet_t mainNet;           // 保存用于单核执行的部分

    // 这里需要一个任务队列 本质上就是 Net 中的复制
    int32_t index_Net[MAX_LAYERS_SEQUENCE];  

    int8_t exitflag;            //退出信号
    int64_t curSize;

    // 这里线程之间的任务分发可以考虑通道或是锁或是C++std中的原子变量
    // 需要注意的是这里的每个线程的将分配的任务并不是对称的
    // 及实际上就是多个单生产者单消费者模型的结合
    // 所以问题的关键是要先确定任务分发的模式和方法
    // 对于主线程来说 mainNet 并不清楚每个 subNet 的情况
    // 或是说如果让 mainNet 来精确的控制每个 subNet 的任务分发代价太大了

    // 读写锁用来实现 主线程与多个子线程之间的任务分发
    // 任务与子线程之间的一一对应的逻辑关系由子线程进行判断选择
    pthread_rwlock_t rwlock;
    pthread_mutex_t mtx_index;
    pthread_mutex_t mtx_curSize; 
    pthread_cond_t cv; 
} threadPool_t;


// int makeThreads(threadPool_t *threadPool);
void *threadFunc(void *arg);

// int create_empty_file(const char *filename, size_t size);

} // namespace end of PyTZone 

#endif