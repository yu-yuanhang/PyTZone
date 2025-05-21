#include "tsops.h"
#include <RunMgr.h>

namespace PyTZone {
namespace core {

TStation::~TStation() {
    LogDebug("~TStation()");
    destory();
}

TStation::TStation(int64_t stn, 
                   int64_t index2, int8_t kpIdx2,
                   int64_t index1, int8_t kpIdx1):
    Layer(TSTATION_TYPE, DN),
    _stn(static_cast<STATION>(int64ToSizeT(stn))),
    _index2(index2),
    _kpIdx2(kpIdx2),
    _index1(INVALID_VALUE),
    _kpIdx1(INVALID_VALUE_U)
{
    LogDebug("TStation(int64_t, int64_t, int64_t)");
    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");

    _layer1 = (_index1 >= 0) ? _layer1 = NETWORK->getLayer(_index1) : nullptr;
    _layer2 = (_index2 >= 0) ? _layer2 = NETWORK->getLayer(_index2) : nullptr;
    // _layer2 = NETWORK->getLayer(_index2);
}

#if LIBTORCH_EXT
TStation::TStation(SerTStationPrePack spp):
    Layer(std::move(std::get<0>(spp))),
    _stn(static_cast<STATION>(Int64TToInt32T(std::get<1>(spp)))),
    _index2(std::get<2>(spp)),
    _kpIdx2(std::get<3>(spp)),
    _index1(std::get<4>(spp)),
    _kpIdx1(std::get<5>(spp))
{
    LogDebug("TStation(SerTStationPrePack)");
    int32_t ret = NETWORK->attachLayer(this, _idx);
    // cout << "ret == " << ret << endl;
    // EXIT_ERROR_CHECK(ret == _idx, false, "error idx from OpsCtx");

    // 这里有个问题就是反序列化的过程中 
    // 算子的顺序是不可控的
    // _layer1 = (_index1 >= 0) ? _layer1 = NETWORK->getLayer(_index1) : nullptr;
    // _layer2 = (_index2 >= 0) ? _layer2 = NETWORK->getLayer(_index2) : nullptr;
}
 
void TStation::operator()(at::Tensor &input) {
    LogDebug("void TStation::operator()(at::Tensor &)");
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
}
void TStation::operator()() {
    LogDebug("void TStation::operator()()");
    NETWORK->registerIdx(_idx);
}
void TStation::attach(at::Tensor &input) {
    LogDebug("void TStation::attach(at::Tensor &) : idx = %d : ", _idx);
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
    // return this;
}
void TStation::attach() {
    LogDebug("void TStation::attach() : idx = %d : ", _idx);
    NETWORK->registerIdx(_idx);
    // return this;
}
void TStation::paramLoad(const string &str, const at::Tensor &data) {
    LogDebug("TStation::Load_param(const std::string &, const at::Tensor &)");
}

const SerTStationPrePack &TStation::makePack() {
    LogDebug("makePack() for TStation");
    SerTStationPrePack *pspp = new SerTStationPrePack(
        std::move(makeLPack()),
        static_cast<int64_t>(_stn),
        _index2, _kpIdx2, _index1, _kpIdx1);
    return *pspp;
}

void TStation::initialize(const at::Tensor &data) {
    LogDebug("TStation::initialize(const at::Tensor &)");
    subIndex();
    // int8_t keepIn = _layer2->getKeepIn();
    // int8_t keepOut = _layer2->getKeepOut();
    // if ((keepIn && _kpIdx2) || (keepOut && !_kpIdx2)) 
    // _inputs = _outputs = (_kpIdx2) ? _layer2->getInputs() : _layer2->getOutputs();

    // int64_t *dims_tmp = (_kpIdx2) ? _layer2->getDmis_t() : _layer2->getOutDmis_t();
    // memcpy(_dims, dims_tmp, MAX_CONV_DIMENSIONS * FLOAT_SIZE);
    // memcpy(_out_dims, dims_tmp, MAX_CONV_DIMENSIONS * FLOAT_SIZE);
}
#endif // LIBTORCH_EXT

void TStation::print() const {
    cout << "!!!!!!!!!!!!!! print() :  TStation !!!!!!!!!!!!!!" << endl \
         << "idx            : " << (getIdx()) << endl \
         << "type           : " << _type << endl \
         << "keepIn         : " << ((_keepIn) ? "ture" : "false") << endl \
         << "keepOut        : " << ((_keepOut) ? "ture" : "false") << endl \
         << "stn            : " << _stn << endl \
         << "index2         : " << _index2 << endl \
         << "index1         : " << _index1 << endl;
    //      << "inputs         : " << _inputs << endl \ 
    //      << "outputs        : " << _outputs << endl;

    // cout << "dims           : ";
    // for (int64_t i = 1; i <= _dims[0]; ++i) cout << _dims[i] << " ";
    // cout << endl;

    // cout << "out_dims       : ";
    // for (int64_t i = 1; i <= _out_dims[0]; ++i) cout << _out_dims[i] << " ";
    // cout << endl;

    return;
}

#if TORCHZONE
void TStation::make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION) const {
    LogDebug("make_layer_ext_CA(TEEC_INVITATION_T * = nullptr) : idx = %d", _idx);
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);
    make_tsops_ca(invitation,
                  static_cast<uint32_t>(_type),
                  _idx, 
                  static_cast<uint32_t>(_stn),
                  Int64TToINTTA(_index2),
                  _kpIdx2,
                  Int64TToINTTA(_index1),
                  _kpIdx1);
}
#endif

} // namespace end of core 
} // namespace end of PyTZone
