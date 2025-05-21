#ifndef __RUNMGR_H__
#define __RUNMGR_H__

#include <RunMgr/all.h>
#include <generic/Singleton.h>
#include <network.h>

#include <thread/threadPool.h>

namespace PyTZone {

// namespace core {
//     class network;  // 作为前向声明
// } // namespace end of core



using core::network;
using core::Layer;

enum NET_TYPE {
    MAIN_NET = 400
};

// self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
// self.conv2_ca = test_ca.Conv2d_ca
// self.pool = nn.MaxPool2d(2, 2)
// self.pool_ca = test_ca.MaxPool2d_ca
// self.fc = nn.Linear(32 * 7 * 7, 10)

// # torch.Size([1, 16, 14, 14])
// x = self.conv2_ca(x)
// # torch.Size([1, 32, 14, 14])
// x = test_ca.relu_ca(x)
// # torch.Size([1, 32, 14, 14])
// x = self.pool_ca(x)

class RunMgr {
public:
    // int32_t attachNet(void *pNet, NET_TYPE netType = MAIN_NET);

public:
    RunMgr();
    ~RunMgr();

    int setThreadSize(size_t threadSize);
    void setTS_ConvMax();
    void setTS_ConvMin();
    size_t makeThreads();
    inline size_t getThreadSize() const {return _threadPool.threadSize;}
    void print() const;

    void initThreadPool();  
    void resetMSNet(threadPool_t &threadPool);
#if TORCHZONE
    void initTeeInv();
    void destoryTeeInv();
    inline TEEC_INVITATION_T *getTeecInv() {return &TEEC_INVITATION;}
#endif
private:
    threadPool_t _threadPool;
#if TORCHZONE
    TEEC_INVITATION_T TEEC_INVITATION;
#endif
};

#define RUNMGR Singleton<RunMgr>::getInstance()
void setupNet();
// void setThreadPool();


} // namespace end of PyTZone  

#endif

