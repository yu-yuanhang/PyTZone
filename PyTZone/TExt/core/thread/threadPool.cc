#include "threadPool.h"
#include <ops/Conv.h>
#include <ops/Linear.h>
#include <ops/NormBatch.h>
#include <ops/Pool.h>
#include <ops/Activ.h>
#include <ops/tsops.h>

using namespace PyTZone::core;

namespace PyTZone {

// int makeThreads(threadPool_t *threadPool) {return 0;}

void *threadFunc(void *arg) {
    pthread_t tid = pthread_self();
    cout << "Thread " << tid << " has been created." << endl;
    // 这里在线程启动之前 pthreadPool 应该已经初始化完成
    threadPool_t *pthreadPool = (threadPool_t *)arg;

    // 获取 subNet 
    pthread_mutex_lock(&pthreadPool->mtx_index);
    Node<subNet_t> *pNet = pthreadPool->subNetArr.get_ptr();
    pthread_mutex_unlock(&pthreadPool->mtx_index);

    while (true) {

       // ...... 


    }

    return nullptr;
}

void Layer::updateNet_thr(struct subNet_s &subNet) {
    subLayer_t *pl = new subLayer_t();
    pl->idx = subNet.num++;

    subNet.inputs = (_inputs > subNet.inputs ? pl->inputs = _inputs : subNet.inputs);
    subNet.output = (_outputs > subNet.output ? pl->output = _outputs : subNet.output);
    subNet.inoutSize = (subNet.inputs > subNet.output ? subNet.inputs : subNet.output);
    subNet.workspace_size = (_workspace_size > subNet.workspace_size ? pl->workspace_size = _workspace_size : subNet.workspace_size);

    pl->weightSize = _nweights;
    pl->biaseSize = _nbiases;
    // mainNet 中偏移量应该都是 0 在 struct 构造时已经初始化

    // add Node in List<Net>
    subNet.index[_idx] = pl->idx;
    subNet.subLayerArr.push_behind(pl); pl = nullptr;
    return;    
}

void ConvNd::updateNet_thr(List<struct subNet_s> &subNet, size_t threadSize) {  
    // ......
    return;
}
void Linear::updateNet_thr(List<struct subNet_s> &subNet, size_t threadSize) {

}
void NormNd::updateNet_thr(List<struct subNet_s> &subNet, size_t threadSize) {

}

}   // end of PyTZone