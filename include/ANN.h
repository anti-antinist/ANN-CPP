#pragma once
#include<vector>
#include<string>

struct NeuronID {
    int l = 0;
    int n = 0;
    NeuronID(int l, int n)
        : l(l), n(n) {};
    NeuronID() = default;
    bool operator==(NeuronID n) {
        if (n.l == this->l && n.n == this->n)
            return true;
        return false;
    }
};

template<typename type>
class ANN {
  private:

    struct NEURON;
    struct LAYER;
    struct ResidualWeight;
    void initializeshit(std::vector<int> &layern, std::vector<std::pair<NeuronID, NeuronID>> ResWeights);
    std::vector<type> costvec(std::vector<type> &target, int l);
    inline static std::vector<LAYER> layers;
    static NEURON &IDtoN(NeuronID nID);
    void SetResidualWeight(NeuronID from, NeuronID to, type weight);
    type (*actfuncHID)(type in);
    type (*actfuncOUT)(type in);
  public:

    ANN(std::vector<int> &layern, std::vector<std::pair<NeuronID, NeuronID>> &ResWeights, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type));
    ANN(std::string filename, bool isBIN, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type));
    ANN() = default;
    ~ANN();
    type costavg(std::vector<type> &target);
    std::vector<type> forwardpropagate(std::vector<type> &input);
    void deserializecsv(std::string filename);
    void serializecsv(std::string filename);
    void deserializebin(std::string filename);
    void serializebin(std::string filename);
    void resetStructure(std::vector<int> &layern, std::vector<std::pair<NeuronID, NeuronID>> &ResWeights);
    template<typename lr_type>
    void backpropagate(std::vector<type> &input, std::vector<type> target, lr_type learn_rate, bool learn_rate_safety);
    template<typename lr_type>
    void batchbackpropagate(std::vector<std::vector<type>> &input, std::vector<std::vector<type>> &target, lr_type learn_rate, bool learn_rate_safety);

    template<typename TYPE_ANN_TRAINER> friend class ANN_TRAINER;
};

template<typename type>
ANN<type>::ANN(std::vector<int> &layern, std::vector<std::pair<NeuronID, NeuronID>> &ResWeights, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type)) {
    initializeshit(layern, ResWeights);
    actfuncHID = actfuncHIDp;
    actfuncOUT = actfuncOUTp;
}

template<typename type>
ANN<type>::ANN(std::string filename, bool isBIN, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type)) {
    if (isBIN) {
        deserializebin(filename);
    }
    else {
        deserializecsv(filename);
    }
    actfuncHID = actfuncHIDp;
    actfuncOUT = actfuncOUTp;
}

template<typename type>
ANN<type>::~ANN() {
    layers.clear();
}

template<typename type>
void ANN<type>::initializeshit(std::vector<int> &layern, std::vector<std::pair<NeuronID, NeuronID>> ResWeights) {
    assert(layern.size() >= 2);
    layers.resize(layern.size());
    for (int i = 0; i < layers.size(); i++) {
        if (i == 0) {
            layers[i].init(layern[i], nullptr);
            continue;
        } else {
            layers[i].init(layern[i], &(*(layers.begin() + i - 1)));
            continue;
        }
    }
    for (int i = 0; i < layers.size(); i++) {
        if (i != layers.size() - 1) {
            for (auto &j : layers[i].neurons) {
                j.initializeweights(&layers[i + 1]);
            }
        }
    }
    for (auto &r : ResWeights) {
        SetResidualWeight(r.first, r.second, rand() / ((double)RAND_MAX * 1.0f) - 1.0f);
    }
}

template<typename type>
std::vector<type> ANN<type>::costvec(std::vector<type> &target, int l) {
    std::vector<type> out;
    for (int n = 0; n < layers[l].neurons.size(); n++) {
        out.push_back(layers[l].neurons[n].activation - target[n]);
    }
    return out;
}
template<typename type>
typename ANN<type>::NEURON &ANN<type>::IDtoN(NeuronID nID) {
    return *&*(layers[nID.l].neurons.begin() + nID.n);
}

template<typename type>
type ANN<type>::costavg(std::vector<type> &target) {
    assert(layers.back().neurons.size() == target.size());
    type cost = 0.0f;
    for (int i = 0; i < layers.back().neurons.size(); i++) {
        cost += layers.back().neurons[i].activation - target[i];
    }
    cost = cost;
    cost /= layers.back().neurons.size();
    return cost;
}

template<typename type>
std::vector<type> ANN<type>::forwardpropagate(std::vector<type> &input) {
    assert(input.size() == layers[0].neurons.size());
    for (auto &n : layers[0].neurons) {
        n.activation = input[n.currentID] + n.bias;
    }
    for (int i = 1; i < layers.size() - 1; i++) {
        for (auto &n : layers[i].neurons) {
            n.calculateActivation(*actfuncHID);
        }
    }
    for (auto &n : layers.back().neurons) {
        n.calculateActivation(*actfuncOUT);
    }
    std::vector<type> out;
    for (auto &n : layers.back().neurons) {
        out.push_back(n.activation);
    }
    return out;
}

template<typename type> 
struct ANN<type>::NEURON {
    type bias = 0.0f;
    int currentID = 0;
    type activation = 0.0f;
    std::vector<type> outweights;
    LAYER *previouslayer = nullptr;
    std::vector<ResidualWeight> resWeights;
    NEURON(LAYER &prev);
    NEURON() = default;
    ~NEURON();
    void calculateActivation(type (&actfunc)(type));
    void initializeweights(LAYER *next);
};

template<typename type> 
ANN<type>::NEURON::NEURON(LAYER &prev) {
    previouslayer = &prev;
}

template<typename type> 
ANN<type>::NEURON::~NEURON() {
    outweights.clear();
    previouslayer = nullptr;
}

template<typename type> 
void ANN<type>::NEURON::calculateActivation(type (&actfunc)(type)) {
    activation = 0.0f;
    for (auto &i : previouslayer->neurons) {
        activation += i.outweights[currentID] * i.activation + bias;
    }
    for (auto &r : resWeights) {
        activation += ANN<type>::IDtoN(r.from).activation * r.weight + bias;
    }
    activation = actfunc(activation);
}

template<typename type> 
struct ANN<type>::LAYER {
  public:

    std::vector<NEURON> neurons;
    void init(int n, LAYER *prev);
    ~LAYER();
};

template<typename type> 
void ANN<type>::NEURON::initializeweights(LAYER *next) {
    srand(std::time(NULL));
    outweights.resize(next->neurons.size(), 1.0f);
    for (type &i : outweights)
        i = rand() / ((double)RAND_MAX * 1.0f) - 1.0f;
    bias = rand() / ((double)RAND_MAX * 1.0f) - 1.0f;
    next = nullptr;
}

template<typename type>
void ANN<type>::LAYER::init(int n, LAYER *prev) {
    neurons.resize(n, NEURON(*prev));
    int x = 0;
    for (auto &i : neurons) {
        i.currentID = x;
        x++;
    }
    prev = nullptr;
}

template<typename type>
ANN<type>::LAYER::~LAYER() {
    neurons.clear();
}

template<typename type> 
struct ANN<type>::ResidualWeight {
    NeuronID from;
    type weight = 0.0f;
    ResidualWeight(NeuronID fromp, type weightp);
};

template<typename type> 
ANN<type>::ResidualWeight::ResidualWeight(NeuronID fromp, type weightp) {
    from = fromp;
    weight = weightp;
}