#include "RunMgr.h"

using std::cout;
using std::endl;

#if 0
int main(int argc, char *argv[]) {

    cout << "hello" << endl;
    int number = 10;
    number = true;
    cout << number << endl;
    // LogDebug("this is an error message!");
    // LogInfo("this is an error message!");
    // LogWarn("this is an error message!");
    // LogError("this is an error message!");
    return 0;
}
#endif 

namespace PyTZone {

using core::Flag_TorchZone;

#if TORCHZONE
void initTee() {
    RUNMGR->initTeeInv();
}
void destoryTee() {
    RUNMGR->destoryTeeInv();
}
void RunMgr::initTeeInv() {
    LogDebug("RunMgr::initTeeInv()");
    prepare_tee_session(&TEEC_INVITATION);
    // alloc_shareMem(1024 * 1024);
}
void RunMgr::destoryTeeInv() {
    LogDebug("RunMgr::destoryTeeInv()");
    terminate_tee_session(&TEEC_INVITATION);
}
#endif
RunMgr::RunMgr():
    _threadPool()
{
    LogDebug("RunMgr()");
}

RunMgr::~RunMgr() {
    // printf("RunMgr::~RunMgr()\n");
    LogDebug("~RunMgr()");
#if TORCHZONE 
    // terminate_tee_session();
#endif
    // 回收子线程
    if (0 == _threadPool.threadSize) {  // 单核心

    } else {
        for (size_t i = 0; i < _threadPool.threadSize; i++) {
            int ret = pthread_join(_threadPool.tidArr[i], NULL);
            EXIT_ERROR_CHECK(ret, -1, "pthread_join");
        }
    }
}

int RunMgr::setThreadSize(size_t threadSize) {
    // 这里理论上要做基本的合理性检查
    _threadPool.threadSize = threadSize;
    return 0;
}

void RunMgr::setTS_ConvMax() {
    _threadPool.threadSize = NETWORK->get_ConvMax_fromThr();
}
void RunMgr::setTS_ConvMin() {
    _threadPool.threadSize = NETWORK->get_ConvMin_fromThr();
}

size_t RunMgr::makeThreads() {
    LogDebug("RunMgr::makeThreads(...)");
    if (0 == _threadPool.threadSize) return 0;
    _threadPool.tidArr.reset(new pthread_t[_threadPool.threadSize]);
    for (size_t i = 0; i < _threadPool.threadSize; ++i) {_threadPool.tidArr[i] = pthread_t();}
    _threadPool.exitflag = 0;

    // 初始化 threadPool
    // ...... 初始化过程中 subNetArr list 中的 ptr 必须指向 head

    // 初始化任务队列
    // .......
    
    for(size_t i = 0; i< _threadPool.threadSize; i++) {
        int ret = pthread_create(&(_threadPool.tidArr[i]), NULL, threadFunc, &_threadPool);
        EXIT_ERROR_CHECK(ret, -1, "pthread_create");
    }
    return 0;
}

void RunMgr::print() const {
    _threadPool.print();
}

void setupNet() {
    NETWORK->initLayerPtrs();
if (Flag_TorchZone) {
#if TORCHZONE
    if (0 == RUNMGR->getThreadSize()) { // 单核
        NETWORK->makeNet_CA();
    } else {
        // ......
    } 
    return;
// #else 
#endif
}
    // std::cout << "setupNet : TORCHZONE not define" << std::endl;
    NETWORK->makeNet();
// #endif
}

void RunMgr::initThreadPool() {
    LogDebug("RunMgr::initThreadPool(...)");
    // 如果没有设置 threadSize 需要自动判断 threadSize 的合适大小
    // 在不考虑 pool 操作的情况下
    // 一般 threadSize 的大小依赖于 Conv 的卷积核数量或是 FullConn 的输出尺寸
    // 两者之间一般考虑前者

    // 在设计上原则上对于每一个 Layer 来说是否需要调用 threadPool 都需要由开发者指定
    // 但是目前实验阶段都是自动生成 Layer 还没有设计对应的数据成员
    // setTS_ConvMax();
    // setTS_ConvMin();
    _threadPool.threadSize = 4;
    
    // init mainNet

    // init subNetArr
}

void RunMgr::resetMSNet(threadPool_t &threadPool) {
    LogDebug("RunMgr::resetMainNet(threadPool_t &)");
    // 这里的关键问题是初始化阶段 net 中 
    // layer 之间的前后关系和执行顺序之间并没有直接或间接联系
    // 因此对于 batchnorm 这样非全依赖但是又具有权重数据的情况
    // 需要提前确定前一层的数据

    // 但是因为 batchnorm 依据 in_c 来进行划分
    // in_c 在数值上等价于 Conv 的 num 等价于 linear 的 out_c 
    // 因此理论上说只要保证划分策略不变就行

    int32_t idx = 0;
    int32_t layerThr = -1;
    size_t threadSize =  threadPool.threadSize;
    
    threadPool.mainNet.subLayerArr.reset_ptr(); 

    Layer *l = nullptr;
    while (true) {
        layerThr = NETWORK->getLayerThr(idx++);
        l = NETWORK->getLayer(idx);
        if (0 == layerThr) {    // nainNet
            l->updateNet_thr(threadPool.mainNet);
            continue;
        }
        if (2 == layerThr) {    // subNet
            // CONV_TYPE FCONNECTED_TYPE
            l->updateNet_thr(threadPool.subNetArr, threadPool.threadSize);
            // if (CONV_TYPE == l->getType()) {}
            // if (FCONNECTED_TYPE == l->getType()) {}
            continue;
        }
        if (1 == layerThr) {
            // BATCHNORM_TYPE 
            // batchnorm in_c == num of Conv 

        }
        if (-1 == layerThr) break;

    }
    return;
}

// int32_t RunMgr::attachNet(void *pNet, NET_TYPE netType) {
//     RET_ERROR_CHECK(pNet, nullptr, "pNet == nullptr", -1);

//     // 设计原则 MainNet 即主网络不能被覆盖
//     if (MAIN_NET == netType && _isMNet) {
//         LogDebug("mainNet will be overwritten !!!");
//         return -1;
//     }

//     switch(netType) {
//         case MAIN_NET: {
//             _mainNet = (Singleton<network> *)pNet;
//             _isMNet = true;
//             break;
//         }
//         // ......

//         default: {}
//     }

//     return 0;
// }

} // namespace end of PyTZone  

