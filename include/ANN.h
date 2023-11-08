#pragma once
#include<vector>

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