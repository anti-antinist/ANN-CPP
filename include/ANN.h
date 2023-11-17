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
    int l = 0;
    int n = 0;
    NeuronID(int l, int n)
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
    std::vector<type> costvec(std::vector<type> &target, int l);
    inline static std::vector<LAYER> layers;
    static NEURON &IDtoN(NeuronID nID);
    void SetResidualWeight(NeuronID from, NeuronID to, type weight);
    inline static type (*actfuncHID)(type in);
    inline static type (*actfuncOUT)(type in);
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
    void deleteNeuron(int lID);
    void addNeuron(int lID);
    void deleteLayer(int lID);
    void addLayer(int lID, int s);

    template<typename TYPE_ANN_TRAINER> friend class ANN_TRAINER;
};

template<typename type>
ANN<type>::ANN(std::vector<int> &layern, std::vector<std::pair<NeuronID, NeuronID>> &ResWeights, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type)){
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
    for (int i = 0; i < layers.size(); i++){
        layers[i].init(layern[i], i);
    }
    for (int i = 0; i < layers.size()-1; i++){
        for (auto &j : layers[i].neurons){
            j.initializeweights(i + 1);
        }
    }
    for (auto &r : ResWeights){
        SetResidualWeight(r.first, r.second, /*rand() / ((double)RAND_MAX * 1.0f) - 1.0f*/1.0f);
    }
}

template<typename type>
std::vector<type> ANN<type>::costvec(std::vector<type> &target, int l){
    std::vector<type> out;
    for (int n = 0; n < layers[l].neurons.size(); n++){
        out.push_back(layers[l].neurons[n].activation - target[n]);
    }
    return out;
}

template<typename type>
typename ANN<type>::NEURON &ANN<type>::IDtoN(NeuronID nID){
    return *&*(layers[nID.l].neurons.begin() + nID.n);
}

template<typename type>
type ANN<type>::costavg(std::vector<type> &target){
    assert(layers.back().neurons.size() == target.size());
    type cost = 0.0f;
    for (int i = 0; i < layers.back().neurons.size(); i++){
        cost += layers.back().neurons[i].activation - target[i];
    }
    cost = cost;
    cost /= layers.back().neurons.size();
    return cost;
}

template<typename type>
std::vector<type> ANN<type>::forwardpropagate(std::vector<type> &input){
    assert(input.size() == layers[0].neurons.size());
    for (auto &n : layers[0].neurons){
        n.activation = input[n.currentID] + n.bias;
    }
    for (int i = 1; i < layers.size() - 1; i++){
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
    int ln = 0;
    int nn = 0;
    std::string tmp;
    std::getline(file, line);
    input.str(line);
    while (std::getline(input, tmp, ',')){
        layerinp.push_back(std::atoi((tmp.c_str())));
    }
    int l = 0, n = 0;
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
        int s_of_r = 0;
        std::getline(input, tmp, ',');
        s_of_r = atoi(tmp.c_str());
        for (int r = 0; r < s_of_r; r++){
            int lf = 0, nf = 0;
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
    for (int i = 0; i < layers.size(); i++)
        line += std::to_string(layers[i].neurons.size()) + ((i < layers.size() - 1) ? "," : "");
    file << line.append("\n");
    for (int l = 0; l < layers.size(); l++){
        for (int n = 0; n < layers[l].neurons.size(); n++){
            line = std::to_string(l) + "," + std::to_string(n);
            if (l < layers.size() - 1)
                for (int w = 0; w < layers[l].neurons[n].outweights.size(); w++){
                    line += "," + std::to_string(layers[l].neurons[n].outweights[w]);
                }
            line += "," + std::to_string(layers[l].neurons[n].bias);
            int s_of_r = layers[l].neurons[n].resWeights.size();
            line += "," + std::to_string(s_of_r);
            for (int r = 0; r < s_of_r; r++){
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
    int layerno;
    file.read(reinterpret_cast<char *>(&layerno), sizeof(int));
    std::vector<int> layern(layerno);
    file.read(reinterpret_cast<char *>(layern.data()), layerno * sizeof(int));

    initializeshit(layern, std::vector<std::pair<NeuronID, NeuronID>>{});
    

    int layerid, neuronid, last = 0;
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
        int s_of_r;
        file.read(reinterpret_cast<char *>(&s_of_r), sizeof(int));
        if (s_of_r)
            layers[layerid].neurons[neuronid].resWeights.clear();
        for (int r = 0; r < s_of_r; r++){
            int l, n;
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
    int layerno = layers.size();
    std::vector<int> layern;

    for (auto &l : layers){
        layern.push_back(l.neurons.size());
    }
    file.write(reinterpret_cast<char *>(&layerno), sizeof(int));
    file.write(reinterpret_cast<char *>(layern.data()), layerno * sizeof(int));

    for (int l = 0; l < layerno; l++){
        for (int n = 0; n < layers[l].neurons.size(); n++){
            int layerid = l, neuronid = n;
            file.write(reinterpret_cast<char *>(&layerid), sizeof(int));
            file.write(reinterpret_cast<char *>(&neuronid), sizeof(int));
            for (int w = 0; w < layers[l].neurons[n].outweights.size(); w++){
                type weight = layers[l].neurons[n].outweights[w];
                file.write(reinterpret_cast<char *>(&weight), sizeof(type));
            }
            type bias = layers[l].neurons[n].bias;
            file.write(reinterpret_cast<char *>(&bias), sizeof(type));
            int s_of_r = layers[l].neurons[n].resWeights.size();
            file.write(reinterpret_cast<char *>(&s_of_r), sizeof(int));
            for (int r = 0; r < s_of_r; r++){
                int lf = layers[l].neurons[n].resWeights[r].from.l, nf = layers[l].neurons[n].resWeights[r].from.n;
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
void ANN<type>::resetStructure(std::vector<int> &layern,std::vector<std::pair<NeuronID, NeuronID>> &ResWeights){
    initializeshit(layern, ResWeights);
}

template<typename type>
template<typename lr_type>
void ANN<type>::backpropagate(std::vector<type> &input, std::vector<type> target, lr_type learn_rate, bool learn_rate_safety){
    forwardpropagate(input);
    type jump_slowdown = 1.0f;
    std::vector<type> delta,prev_delta;
    for(int l = layers.size()-1; l >= 0; l--){
        std::vector<std::pair<ResidualWeight&, type>> res_w;
        if(l == layers.size()-1){
            delta.resize(target.size());
            delta = costvec(target, l);
        }
        else{
            prev_delta.clear();
            prev_delta.resize(delta.size());
            prev_delta = delta;
            delta.resize(layers[l].neurons.size(),0.0f);
            for(int i = 0; i < delta.size(); i++){
                type sum = 0.0f;
                for(int t = 0; t < prev_delta.size(); t++){
                    sum += prev_delta[t]*layers[l].neurons[i].outweights[t];
                }
                for(int r = 0; r < res_w.size(); r++){
                    if(res_w[r].first.from == NeuronID(l,i)){
                        sum += res_w[r].second*res_w[r].first.weight;
                    }
                }
                delta[i] = layers[l].neurons[i].activation*(1-layers[l].neurons[i].activation)*sum;
            }
        }
        for (int n = 0; n < layers[l].neurons.size(); n++){
            for (int r = 0; r < layers[l].neurons[n].resWeights.size(); r++){
                res_w.push_back(std::pair<ResidualWeight &, type>(layers[l].neurons[n].resWeights[r], delta[n]));
            }
        }
        if(l > 0){
            for(int n = 0; n < layers[l-1].neurons.size(); n++){
                for(int o = 0; o < layers[l].neurons.size(); o++){
                    layers[l-1].neurons[n].outweights[o] -= learn_rate * layers[l-1].neurons[n].activation * delta[o];
                }
            }
            for(int n = 0; n < layers[l].neurons.size(); n++){
                for(int r = 0; r < layers[l].neurons[n].resWeights.size(); r++){
                    layers[l].neurons[n].resWeights[r].weight -= learn_rate * IDtoN(layers[l].neurons[n].resWeights[r].from).activation * delta[n];
                }
            }
        }
        for(int b = 0; b < layers[l].neurons.size(); b++){
            layers[l].neurons[b].bias -= learn_rate * delta[b];
        }
    }
}

template<typename type>
void ANN<type>::deleteNeuron(int lID){
    assert(lID >= 0 && lID < layers.size());
    layers[lID].neurons.erase(layers[lID].neurons.end());
}

template<typename type>
void ANN<type>::addNeuron(int lID){
    assert(lID >= 0 && lID < layers.size());
    srand(std::time(NULL));
    layers[lID].neurons.push_back(NEURON());
    if(lID < layers.size()-1){ 
        layers[lID].neurons.back().initializeweights(lID+1);
    }
}

template<typename type>
void ANN<type>::deleteLayer(int lID){
    if(lID > 0 && lID < layers.size()-1){
        std::vector<std::vector<type>> new_w;
        new_w.resize(layers[lID-1].neurons.size(),std::vector<type>(layers[lID+1].neurons.size(),1.0f));
        for(int n = 0; n < layers[lID].neurons.size() && n < new_w.size(); n++){
            for(int o = 0; o < layers[lID].neurons[n].outweights.size(); o++){
                new_w[n][o] = (layers[lID].neurons[n].outweights[o]);
            }
        }
        layers.erase(lID + layers.begin());
        for(int l = lID; l < layers.size(); l++){
            layers[l].curr_l--;
        }
        for(auto& n : layers[lID-1].neurons){
            n.outweights.clear();
            for(auto& w : new_w[n.currentID]){
                n.outweights.push_back(w);
            }
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
void ANN<type>::SetResidualWeight(NeuronID from, NeuronID to, type weight){
    assert(from.l < layers.size() || to.l < layers.size());
    assert(from.n < layers[from.l].neurons.size() || to.n < layers[to.l].neurons.size());
    assert(from.l < to.l);
    layers[to.l].neurons[to.n].resWeights.push_back(ResidualWeight(from, weight));
}

template<typename type> 
struct ANN<type>::NEURON{
    type bias = 0.0f;
    int currentID = 0;
    type activation = 0.0f;
    std::vector<type> outweights;
    std::vector<ResidualWeight> resWeights;
    NEURON() = default;
    ~NEURON();
    void initializeweights(int next);
};

template<typename type> 
ANN<type>::NEURON::~NEURON(){
    outweights.clear();
}

template<typename type> 
void ANN<type>::NEURON::initializeweights(int next){
    outweights.resize(layers[next].neurons.size(), 1.0f);
    for (type &i : outweights)
        i = rand() / ((double)RAND_MAX * 1.0f) - 1.0f;
    bias = rand() / ((double)RAND_MAX * 1.0f) - 1.0f;
}

template<typename type> 
struct ANN<type>::LAYER{
  public:

    std::vector<NEURON> neurons;
    int curr_l = 0;
    void init(int n, int curr);
    void calcActs(type(actfunc)(type));
    ~LAYER();
};

template<typename type>
void ANN<type>::LAYER::init(int n, int curr){
    neurons.resize(n, NEURON());
    curr_l = curr;
    int x = 0;
    for (auto &i : neurons){
        i.currentID = x;
        x++;
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
ANN<type>::LAYER::~LAYER(){
    neurons.clear();
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