#pragma once
#include<vector>

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
struct ANN<type>::LAYER {
  public:

    std::vector<NEURON> neurons;
    void init(int n, LAYER *prev);
    ~LAYER();
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