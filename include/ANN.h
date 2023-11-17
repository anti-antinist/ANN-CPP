#pragma once
#include<vector>
#include<string>
#include<fstream>
#include<ctime>
#include<cassert>
#include<sstream>
#include<utility>
#include<iostream>

struct NeuronID{
    unsigned int l = 0;
    unsigned int n = 0;
    NeuronID(unsigned int l, unsigned int n)
        : l(l), n(n){};
    NeuronID() = default;
    bool operator==(NeuronID n){
        if (n.l == this->l && n.n == this->n)
            return true;
        return false;
    }
};

template<typename type>
class ANN{
    private:

        struct NEURON;
        struct LAYER;
        struct ResidualWeight;
        void initializeshit(std::vector<int> &layern, std::vector<std::pair<NeuronID, NeuronID>> ResWeights);
        std::vector<type> costvec(std::vector<type> &target, unsigned int l);
        inline static std::vector<LAYER> layers;
        static NEURON &IDtoN(NeuronID nID);
        inline static type (*actfuncHID)(type in);
        inline static type (*actfuncOUT)(type in);
    public:

        ANN(const std::vector<int> &layern, const std::vector<std::pair<NeuronID, NeuronID>> &ResWeights, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type));
        ANN(std::string filename, bool isBIN, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type));
        ANN() = default;
        ~ANN();
        type costavg(const std::vector<type> &target);
        std::vector<type> forwardpropagate(const std::vector<type> &input);
        void deserializecsv(std::string filename);
        void serializecsv(std::string filename);
        void deserializebin(std::string filename);
        void serializebin(std::string filename);
        void resetStructure(const std::vector<int> &layern, const std::vector<std::pair<NeuronID, NeuronID>> &ResWeights);
        template<typename lr_type>
        void backpropagate(const std::vector<type> &input, const std::vector<type> &target, lr_type learn_rate, bool learn_rate_safety);
        void deleteNeuron(unsigned int lID);
        void addNeuron(unsigned int lID);
        void deleteLayer(unsigned int lID);
        void addLayer(unsigned int lID, unsigned int s);
        void AddResidualWeight(NeuronID from, NeuronID to, type weight);
        void DeleteResidualWeight(NeuronID from, NeuronID to);

        template<typename TYPE_ANN_TRAINER> friend class ANN_TRAINER;
};

template<typename type>
ANN<type>::ANN(const std::vector<int> &layern, const std::vector<std::pair<NeuronID, NeuronID>> &ResWeights, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type)){
    initializeshit(layern, ResWeights);
    actfuncHID = actfuncHIDp;
    actfuncOUT = actfuncOUTp;
}

template<typename type>
ANN<type>::ANN(std::string filename, bool isBIN, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type)){
    if (isBIN){
        deserializebin(filename);
    }
    else{
        deserializecsv(filename);
    }
    actfuncHID = actfuncHIDp;
    actfuncOUT = actfuncOUTp;
}

template<typename type>
ANN<type>::~ANN(){
    layers.clear();
}

template<typename type>
void ANN<type>::initializeshit(std::vector<int> &layern, std::vector<std::pair<NeuronID, NeuronID>> ResWeights){
    assert(layern.size() >= 2);
    srand(std::time(NULL));
    layers.resize(layern.size());
    for (unsigned int i = 0; i < layers.size(); i++){
        layers[i].init(layern[i], i, (layern.size()-1 == i) ? 0 : layern[i+1]);
    }
    for (auto &r : ResWeights){
        AddResidualWeight(r.first, r.second, 1.0f);
    }
}

template<typename type>
std::vector<type> ANN<type>::costvec(std::vector<type> &target, unsigned int l){
    std::vector<type> out;
    for (unsigned int n = 0; n < layers[l].neurons.size(); n++){
        out.push_back(layers[l].neurons[n].activation - target[n]);
    }
    return out;
}

template<typename type>
typename ANN<type>::NEURON &ANN<type>::IDtoN(NeuronID nID){
    return *&*(layers[nID.l].neurons.begin() + nID.n);
}

template<typename type>
type ANN<type>::costavg(const std::vector<type> &target){
    assert(layers.back().neurons.size() == target.size());
    type cost = 0.0f;
    for (unsigned int i = 0; i < layers.back().neurons.size(); i++){
        cost += layers.back().neurons[i].activation - target[i];
    }
    cost = cost;
    cost /= layers.back().neurons.size();
    return cost;
}

template<typename type>
std::vector<type> ANN<type>::forwardpropagate(const std::vector<type> &input){
    assert(input.size() == layers[0].neurons.size());
    for (auto &n : layers[0].neurons){
        n.activation = input[n.currentID] + n.bias;
    }
    for (unsigned int i = 1; i < layers.size() - 1; i++){
        layers[i].calcActs(*actfuncHID);
    }
    layers.back().calcActs(*actfuncOUT);
    std::vector<type> out;
    for (auto &n : layers.back().neurons){
        out.push_back(n.activation);
    }
    return out;
}

template<typename type>
void ANN<type>::deserializecsv(std::string filename){
    std::ifstream file(filename);
    if (!file.is_open()){
        return;
    }
    std::string line;
    std::stringstream input;
    std::vector<int> layerinp;
    unsigned int ln = 0;
    unsigned int nn = 0;
    std::string tmp;
    std::getline(file, line);
    input.str(line);
    while (std::getline(input, tmp, ',')){
        layerinp.push_back(std::atoi((tmp.c_str())));
    }
    unsigned int l = 0, n = 0;
    initializeshit(layerinp, std::vector<std::pair<NeuronID, NeuronID>>{});
    while (std::getline(file, line)){
        std::stringstream().swap(input);
        input << line;
        std::getline(input, tmp, ',');
        l = atoi(tmp.c_str());
        std::getline(input, tmp, ',');
        n = atoi(tmp.c_str());
        for (auto &i : layers[l].neurons[n].outweights){
            std::getline(input, tmp, ',');
            i = atof(tmp.c_str());
        }
        std::getline(input, tmp, ',');
        layers[l].neurons[n].bias = atof(tmp.c_str());
        unsigned int s_of_r = 0;
        std::getline(input, tmp, ',');
        s_of_r = atoi(tmp.c_str());
        for (unsigned int r = 0; r < s_of_r; r++){
            unsigned int lf = 0, nf = 0;
            std::getline(input, tmp, ',');
            lf = atof(tmp.c_str());
            std::getline(input, tmp, ',');
            nf = atof(tmp.c_str());
            type w = 0.0f;
            std::getline(input, tmp, ',');
            w = atof(tmp.c_str());
            layers[l].neurons[n].resWeights.push_back(ResidualWeight(NeuronID(l, n), w));
        }
    }
    input.clear();
    file.close();
};

template<typename type>
void ANN<type>::serializecsv(std::string filename){
    std::ofstream file(filename);
    if (!file.is_open()){
        return;
    }
    std::string line;
    for (unsigned int i = 0; i < layers.size(); i++)
        line += std::to_string(layers[i].neurons.size()) + ((i < layers.size() - 1) ? "," : "");
    file << line.append("\n");
    for (unsigned int l = 0; l < layers.size(); l++){
        for (unsigned int n = 0; n < layers[l].neurons.size(); n++){
            line = std::to_string(l) + "," + std::to_string(n);
            if (l < layers.size() - 1)
                for (unsigned int w = 0; w < layers[l].neurons[n].outweights.size(); w++){
                    line += "," + std::to_string(layers[l].neurons[n].outweights[w]);
                }
            line += "," + std::to_string(layers[l].neurons[n].bias);
            unsigned int s_of_r = layers[l].neurons[n].resWeights.size();
            line += "," + std::to_string(s_of_r);
            for (unsigned int r = 0; r < s_of_r; r++){
                line += "," + std::to_string(layers[l].neurons[n].resWeights[r].from.l) + "," + std::to_string(layers[l].neurons[n].resWeights[r].from.n) + "," + std::to_string(layers[l].neurons[n].resWeights[r].weight);
            }
            file << line << "\n";
        }
    }
    line.clear();
    file.close();
}

template<typename type>
void ANN<type>::deserializebin(std::string filename){
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()){
        return;
    }
    unsigned int layerno;
    file.read(reinterpret_cast<char *>(&layerno), sizeof(int));
    std::vector<int> layern(layerno);
    file.read(reinterpret_cast<char *>(layern.data()), layerno * sizeof(int));

    initializeshit(layern, std::vector<std::pair<NeuronID, NeuronID>>{});
    

    unsigned int layerid, neuronid, last = 0;
    while (file.read(reinterpret_cast<char *>(&layerid), sizeof(int)) &&
           file.read(reinterpret_cast<char *>(&neuronid), sizeof(int))){
        if (last > layerid)
            break;
        type weight = 0.0f;
        for (auto &w : layers[layerid].neurons[neuronid].outweights){
            file.read(reinterpret_cast<char *>(&weight), sizeof(type));
            w = weight;
        }
        type bias;
        file.read(reinterpret_cast<char *>(&bias), sizeof(type));
        layers[layerid].neurons[neuronid].bias = bias;
        last = layerid;
        unsigned int s_of_r;
        file.read(reinterpret_cast<char *>(&s_of_r), sizeof(int));
        if (s_of_r)
            layers[layerid].neurons[neuronid].resWeights.clear();
        for (unsigned int r = 0; r < s_of_r; r++){
            unsigned int l, n;
            type w;
            file.read(reinterpret_cast<char *>(&l), sizeof(int));
            file.read(reinterpret_cast<char *>(&n), sizeof(int));
            file.read(reinterpret_cast<char *>(&w), sizeof(type));

            layers[layerid].neurons[neuronid].resWeights.push_back(ResidualWeight(NeuronID(l, n), w));
        }
    }

    file.close();
}

template<typename type>
void ANN<type>::serializebin(std::string filename){
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()){
        return;
    }
    unsigned int layerno = layers.size();
    std::vector<int> layern;

    for (auto &l : layers){
        layern.push_back(l.neurons.size());
    }
    file.write(reinterpret_cast<char *>(&layerno), sizeof(int));
    file.write(reinterpret_cast<char *>(layern.data()), layerno * sizeof(int));

    for (unsigned int l = 0; l < layerno; l++){
        for (unsigned int n = 0; n < layers[l].neurons.size(); n++){
            unsigned int layerid = l, neuronid = n;
            file.write(reinterpret_cast<char *>(&layerid), sizeof(int));
            file.write(reinterpret_cast<char *>(&neuronid), sizeof(int));
            for (unsigned int w = 0; w < layers[l].neurons[n].outweights.size(); w++){
                type weight = layers[l].neurons[n].outweights[w];
                file.write(reinterpret_cast<char *>(&weight), sizeof(type));
            }
            type bias = layers[l].neurons[n].bias;
            file.write(reinterpret_cast<char *>(&bias), sizeof(type));
            unsigned int s_of_r = layers[l].neurons[n].resWeights.size();
            file.write(reinterpret_cast<char *>(&s_of_r), sizeof(int));
            for (unsigned int r = 0; r < s_of_r; r++){
                unsigned int lf = layers[l].neurons[n].resWeights[r].from.l, nf = layers[l].neurons[n].resWeights[r].from.n;
                type w = layers[l].neurons[n].resWeights[r].weight;
                file.write(reinterpret_cast<char *>(&lf), sizeof(int));
                file.write(reinterpret_cast<char *>(&nf), sizeof(int));
                file.write(reinterpret_cast<char *>(&w), sizeof(type));
            }
        }
    }
    file.close();
}

template<typename type> 
void ANN<type>::resetStructure(const std::vector<int> &layern, const std::vector<std::pair<NeuronID, NeuronID>> &ResWeights){
    initializeshit(layern, ResWeights);
}

template<typename type>
template<typename lr_type>
void ANN<type>::backpropagate(const std::vector<type> &input, const std::vector<type>& target, lr_type learn_rate, bool learn_rate_safety){
    forwardpropagate(input);
    type jump_slowdown = 1.0f;
    std::vector<type> delta, prev_delta;
    std::vector<std::pair<ResidualWeight&, type>> res_w;
    for(int l = layers.size()-1; l >= 0; l--){
        if(l == layers.size()-1){
            delta.resize(target.size());
            delta = costvec(target, l);
        }
        else{
            prev_delta.resize(delta.size());
            prev_delta = delta;
            delta.resize(layers[l].neurons.size(), 0.0f);
            for(unsigned int i = 0; i < delta.size(); i++){
                type sum = 0.0f;
                for(unsigned int t = 0; t < prev_delta.size(); t++){
                    sum += prev_delta[t]*layers[l].neurons[i].outweights[t];
                }
                for(unsigned int r = 0; r < res_w.size(); r++){
                    if(res_w[r].first.from == NeuronID(l, i)){
                        sum += res_w[r].second*res_w[r].first.weight;
                    }
                }
                delta[i] = layers[l].neurons[i].activation*(1-layers[l].neurons[i].activation)*sum;
                for (unsigned int r = 0; r < layers[l].neurons[i].resWeights.size(); r++){
                    res_w.push_back(std::pair<ResidualWeight &, type>(layers[l].neurons[i].resWeights[r], delta[i]));
                }
            }
        }
        if(l > 0){
            for(unsigned int n = 0; n < layers[l-1].neurons.size(); n++){
                for(unsigned int o = 0; o < layers[l].neurons.size(); o++){
                    layers[l-1].neurons[n].outweights[o] -= learn_rate * layers[l-1].neurons[n].activation * delta[o];
                }
            }
            for(unsigned int n = 0; n < layers[l].neurons.size(); n++){
                for(unsigned int r = 0; r < layers[l].neurons[n].resWeights.size(); r++){
                    layers[l].neurons[n].resWeights[r].weight -= learn_rate * IDtoN(layers[l].neurons[n].resWeights[r].from).activation * delta[n];
                }
            }
        }
        for(unsigned int b = 0; b < layers[l].neurons.size(); b++){
            layers[l].neurons[b].bias -= learn_rate * delta[b];
        }
    }
}

template<typename type>
void ANN<type>::deleteNeuron(unsigned int lID){
    assert(lID >= 0 && lID < layers.size());
    layers[lID].neurons.erase(layers[lID].neurons.end());
}

template<typename type>
void ANN<type>::addNeuron(unsigned int lID){
    assert(lID >= 0 && lID < layers.size());
    srand(std::time(NULL));
    layers[lID].neurons.push_back(NEURON());
    if(lID < layers.size()-1){ 
        layers[lID].neurons.back().initializeweights(lID+1);
    }
}

template<typename type>
void ANN<type>::deleteLayer(unsigned int lID){
    if(lID > 0 && lID < layers.size()-1){
        for(unsigned int n = 0; n < layers[lID-1].neurons.size(); n++){
            if(n < layers[lID].neurons.size()){
                layers[lID-1].neurons[n].outweights.clear();
                layers[lID-1].neurons[n].outweights.resize(layers[lID+1].neurons.size());
                layers[lID-1].neurons[n].outweights = layers[lID].neurons[n].outweights;
            }
            else{
                layers[lID-1].neurons[n].outweights.clear();
                layers[lID-1].neurons[n].outweights.resize(layers[lID+1].neurons.size());                
            }
        }
        layers.erase(lID + layers.begin());
        for(unsigned int l = lID; l < layers.size(); l++){
            layers[l].curr_l--;
        }
    }
    else if(lID == 0){
        layers.erase(layers.begin());
        for(auto& l : layers){
            l.curr_l--;
        }
    }
    else{
        layers.erase(layers.end());
        for(auto& n : layers.back().neurons){
            n.outweights.clear();
        }
    }
}

template<typename type>
void ANN<type>::addLayer(unsigned int lID, unsigned int s){
    layers.insert(lID+layers.begin(), LAYER(s, lID, layers[lID].neurons.size()));
    if(lID > 0 && lID < layers.size()-1){
        for(unsigned int n = 0; n < layers[lID].neurons.size(); n++){
            if(n <= layers[lID-1].neurons.size()-1){
                layers[lID].neurons[n].outweights = layers[lID-1].neurons[n].outweights;
                layers[lID-1].neurons[n].outweights.resize(layers[lID].neurons.size());
                layers[lID-1].neurons[n].initializeweights();
            }
        }
        for(unsigned int l = lID+1; l < layers.size(); l++){
            layers[l].curr_l++;
        }
    }
    else if(lID == 0){
        for(unsigned int l = 0; l < layers.size(); l++){
            layers[l].curr_l++;
        }
    }
    else{
        layers[layers.size()-1].init(s, lID, 0);
        layers[lID-1].init(layers[lID-1].neurons.size(), lID-1, layers[lID].neurons.size());
    }
}

template<typename type> 
void ANN<type>::AddResidualWeight(NeuronID from, NeuronID to, type weight){
    assert(from.l < layers.size() || to.l < layers.size());
    assert(from.n < layers[from.l].neurons.size() || to.n < layers[to.l].neurons.size());
    assert(from.l < to.l);
    layers[to.l].neurons[to.n].resWeights.push_back(ResidualWeight(from, weight));
}

template<typename type>
void ANN<type>::DeleteResidualWeight(NeuronID from, NeuronID to){
    assert(from.l < layers.size() || to.l < layers.size());
    assert(from.n < layers[from.l].neurons.size() || to.n < layers[to.l].neurons.size());
    assert(from.l < to.l);
    for(unsigned int r = 0; r < layers[to.l].neurons[to.n].resWeights.size(); r++){
        if(layers[to.l].neurons[to.n].resWeights[r].from == from){
            layers[to.l].neurons[to.n].resWeights.erase(layers[to.l].neurons[to.n].resWeights.begin()+r);
            break;
        }
    }
}

template<typename type> 
struct ANN<type>::NEURON{
    type bias = 0.0f;
    unsigned int currentID = 0;
    type activation = 0.0f;
    std::vector<type> outweights;
    std::vector<ResidualWeight> resWeights;
    NEURON() = default;
    ~NEURON();
    void initializeweights();
    friend struct LAYER;
};

template<typename type> 
ANN<type>::NEURON::~NEURON(){
    outweights.clear();
}

template<typename type> 
void ANN<type>::NEURON::initializeweights(){
    for (type &i : outweights)
        i = rand() / ((double)RAND_MAX * 1.0f) - 1.0f;
    bias = rand() / ((double)RAND_MAX * 1.0f) - 1.0f;
}

template<typename type> 
struct ANN<type>::LAYER{
  public:

    std::vector<NEURON> neurons;
    unsigned int curr_l = 0;
    LAYER(unsigned int n, unsigned int curr, unsigned int next_s);
    LAYER() = default;
    ~LAYER();
    void init(unsigned int n, unsigned int curr, unsigned int next_s);
    void calcActs(type(actfunc)(type));
};

template<typename type>
ANN<type>::LAYER::LAYER(unsigned int n, unsigned int curr, unsigned int next_s){
    init(n, curr, next_s);
}

template<typename type>
ANN<type>::LAYER::~LAYER(){
    neurons.clear();
}

template<typename type>
void ANN<type>::LAYER::init(unsigned int n, unsigned int curr, unsigned int next_s){
    neurons.resize(n, NEURON());
    curr_l = curr;
    unsigned int x = 0;
    for (auto &i : neurons){
        i.currentID = x;
        x++;
        if(curr < layers.size()-1){
            i.outweights.resize(next_s);
            i.initializeweights();
        }
    }
}

template<typename type>
void ANN<type>::LAYER::calcActs(type(actfunc)(type)){
    for(auto& n : layers[curr_l].neurons){
        n.activation = 0.0f;
        for(auto& np : layers[curr_l-1].neurons){
            n.activation += np.outweights[n.currentID]*np.activation;
        }
        for(auto& r : n.resWeights){
            n.activation += IDtoN(r.from).activation * r.weight;
        }
        n.activation = actfunc(n.activation+n.bias);
    }
}

template<typename type> 
struct ANN<type>::ResidualWeight{
    NeuronID from;
    type weight = 0.0f;
    ResidualWeight(NeuronID fromp, type weightp);
};

template<typename type> 
ANN<type>::ResidualWeight::ResidualWeight(NeuronID fromp, type weightp){
    from = fromp;
    weight = weightp;
}

template<typename type> 
class GeneticEvolution{
    private:
        std::vector<ANN<type>*> networks;
};