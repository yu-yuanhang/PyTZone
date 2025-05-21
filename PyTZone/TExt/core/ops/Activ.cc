#include "Activ.h"
#include <RunMgr.h>
#include <math.h>

namespace PyTZone {
namespace core {

unordered_map<string, ACTIVATION> ActivMap = {
    {"NONE", NONE_ACTIV},
    {"LOGISTIC", LOGISTIC_ACTIV},
    {"RELU", RELU_ACTIV},
    {"RELU6", RELU6_ACTIV},
    {"RELIE", RELIE_ACTIV},
    {"LINEAR", LINEAR_ACTIV},
    {"RAMP", RAMP_ACTIV},
    {"TANH", TANH_ACTIV},
    {"PLSE", PLSE_ACTIV},
    {"LEAKY", LEAKY_ACTIV},
    {"ELU", ELU_ACTIV},
    {"LOGGY", LOGGY_ACTIV},
    {"STAIR", STAIR_ACTIV},
    {"HARDTAN", HARDTAN_ACTIV},
    {"LHTAN", LHTAN_ACTIV},
    {"SELU", SELU_ACTIV}
};

Activ::~Activ() {
    LogDebug("~Activ()");
    destory();
}

Activ::Activ(int64_t activ):
    Layer(ACTIV_TYPE, DN),
    _activ(NONE_ACTIV)
{
    LogDebug("Activ(int64_t)");

    if (activ >= LOGISTIC_ACTIV && activ <= SELU_ACTIV) _activ = static_cast<ACTIVATION>(Int64TToInt32T(activ));
    else LogError("error input for activ");

    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
    setActivate();
}

Activ::Activ(std::string activ):
    Layer(ACTIV_TYPE, DN),
    _activ(NONE_ACTIV)   // NONE_ACTIV
{
    LogDebug("Activ(int64_t)");

    auto it = ActivMap.find(activ);
    if (it == ActivMap.end())
        LogError("error input for activ");
    else {
        _activ = ActivMap[activ];
    }

    _idx = NETWORK->attachLayer(this);
    EXIT_ERROR_CHECK(_idx, INT32_MAX, "error idx from attachLayer");
    setActivate();
}
#if LIBTORCH_EXT
Activ::Activ(SerActivPrePack spp):
    Layer(std::move(std::get<0>(spp))),
    _activ(static_cast<ACTIVATION>(Int64TToInt32T(std::get<1>(spp))))
{
    LogDebug("Activ(SerActivPrePack)");
    int32_t ret = NETWORK->attachLayer(this, _idx);
    // cout << "ret == " << ret << endl;
    // EXIT_ERROR_CHECK(ret == _idx, false, "error idx from OpsCtx");
    setActivate();
}


void Activ::operator()(at::Tensor &input) {
    LogDebug("void Activ::operator()(at::Tensor &)");
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
}
void Activ::operator()() {
    LogDebug("void Activ::operator()()");
    NETWORK->registerIdx(_idx);
}
void Activ::attach(at::Tensor &input) {
    LogDebug("void Activ::attach(at::Tensor &) : idx = %d : ", _idx);
    NETWORK->resetInput(input);
    NETWORK->resetIndex();
    NETWORK->registerIdx(_idx);
    // return this;
}
void Activ::attach() {
    LogDebug("void Activ::attach() : idx = %d : ", _idx);
    NETWORK->registerIdx(_idx);
    // return this;
}
void Activ::paramLoad(const string &str, const at::Tensor &data) {
    LogDebug("Activ::Load_param(const std::string &, const at::Tensor &)");
    if (!strcmp(str.c_str(), "inputs")) {
        initialize(data);
    }
}

const SerActivPrePack &Activ::makePack() {
    LogDebug("makePack() for Activ");
    SerActivPrePack *pspp = new SerActivPrePack(
        std::move(makeLPack()),
        static_cast<int64_t>(_activ));
    return *pspp;
}

void Activ::initialize(const at::Tensor &data) {
    LogDebug("Activ::initialize(const at::Tensor &)");
    // EXIT_ERROR_CHECK(data.dtype() == at::kFloat, false, "type in data if not FLOATCA");
    // setBatch(data);
    int64_t dims_tmp[MAX_CONV_DIMENSIONS] = {0};
    get_tensor_dimensions(data, dims_tmp);
    _batch = getDimsIdx(dims_tmp, 1);
    _outputs = _inputs = (static_cast<int64_t>(data.numel())) / _batch;
    setDims(_dims, dims_tmp, dims_tmp[0] - 1);     // _dims
    setDims(_out_dims, dims_tmp, dims_tmp[0] - 1);
    updateNet();
}
#endif // LIBTORCH_EXT

void Activ::print() const {
    cout << "!!!!!!!!!!!!!! print() :  Activ !!!!!!!!!!!!!!" << endl \
         << "idx            : " << (getIdx()) << endl \
         << "type           : " << _type << endl \
         << "keepIn         : " << ((_keepIn) ? "ture" : "false") << endl \
         << "keepOut        : " << ((_keepOut) ? "ture" : "false") << endl \
         << "batch          : " << _batch << endl \
         << "inputs         : " << _inputs << endl \
         << "outputs        : " << _outputs << endl;

    cout << "dims           : ";
    for (int64_t i = 1; i <= _dims[0]; ++i) cout << _dims[i] << " ";
    cout << endl;

    cout << "out_dims       : ";
    for (int64_t i = 1; i <= _out_dims[0]; ++i) cout << _out_dims[i] << " ";
    cout << endl;

    cout << "Activ          : "; 
    if (_activ) {
        for (const auto &pair : ActivMap) {
            if (pair.second == _activ) {
                cout << pair.first << endl;
                break;
            }
        }
    }
    return;
}

#if TORCHZONE
void Activ::make_layer_ext_CA(TEEC_INVITATION_T *TEEC_INVITATION) const {
    LogDebug("make_layer_ext_CA(TEEC_INVITATION_T * = nullptr) : idx = %d", _idx);
    TEEC_INVITATION_T *invitation = (nullptr == TEEC_INVITATION ? RUNMGR->getTeecInv() : TEEC_INVITATION);
    make_activ_ca(invitation,
                  static_cast<uint32_t>(_type),
                  _idx, 
                  static_cast<uint32_t>(_activ));
}
#endif

} // namespace end of core
} // namespace end of PyTZone