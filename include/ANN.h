#ifndef ANN_H
    #define ANN_H
    #include<vector>
    #include<string>
    #include<fstream>
    #include<ctime>
    #include<cassert>
    #include<sstream>
    #include<utility>

    struct NeuronID{
        unsigned int l = 0;
        unsigned int n = 0;
        NeuronID(unsigned int l, unsigned int n) : l(l), n(n){};
        NeuronID() = default;
        bool operator==(NeuronID n){
            return n.l == this->l && n.n == this->n;
        }
    };

    template<typename type>
    class ANN{

        private:

            struct NEURON;
            struct LAYER;
            struct ResidualWeight;
            void initialize(const std::vector<unsigned int>& layern, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights);
            std::vector<type> costvec(const std::vector<type>& target, unsigned int l);
            std::vector<LAYER> layers;
            NEURON& IDtoN(NeuronID nID);
            type (*actfuncHID)(type in);
            type (*actfuncOUT)(type in);

        public:

            ANN(const std::vector<unsigned int>& layern, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type));
            ANN(std::string filename, bool isBIN, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type));
            ANN(const ANN<type>& net);
            ANN() = default;
            ~ANN();
            type costavg(const std::vector<type>& target);
            std::vector<type> forwardpropagate(const std::vector<type>& input);
            void deserializecsv(std::string filename);
            void serializecsv(std::string filename);
            void deserializebin(std::string filename);
            void serializebin(std::string filename);
            void resetStructure(const std::vector<int>& layern, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights);
            void backpropagate(const std::vector<type>& input, const std::vector<type>& target, type learn_rate, bool learn_rate_safety);
            void deleteNeuron(unsigned int lID);
            void addNeuron(unsigned int lID);
            void deleteLayer(unsigned int lID);
            void addLayer(unsigned int lID, unsigned int s);
            void DeleteResidualWeight(NeuronID from, NeuronID to);
            void AddResidualWeight(NeuronID from, NeuronID to, type weight);

            friend struct NEURON;
            friend struct LAYER;
            template<typename TYPE_EVO_TRAINER> friend class EVO_TRAINER;
    };

    template<typename type>
    ANN<type>::ANN(const std::vector<unsigned int>& layern, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights, type (*actfuncHIDp)(type), type (*actfuncOUTp)(type)){
        initialize(layern, ResWeights);
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
    ANN<type>::ANN(const ANN<type>& net){
        actfuncHID = net.actfuncHID;
        actfuncOUT = net.actfuncOUT;
        for(unsigned int l = 0; l < net.layers.size(); l++){
            if(l < net.layers.size()-1){
                layers.push_back(LAYER(net.layers[l].neurons.size(), l, net.layers[l+1].neurons.size(), this));
            }
            else{
                layers.push_back(LAYER(net.layers[l].neurons.size(), l, 0, this));
            }
            for(unsigned int n = 0; n < layers[l].neurons.size(); n++){
                for(unsigned int o = 0; o < layers[l].neurons[n].outweights.size(); o++){
                    layers[l].neurons[n].outweights[o] = net.layers[l].neurons[n].outweights[o];
                }
                for(unsigned int r = 0; r < net.layers[l].neurons[n].resWeights.size(); r++){
                    layers[l].neurons[n].resWeights.push_back(net.layers[l].neurons[n].resWeights[r]);
                }
                layers[l].neurons[n].bias = net.layers[l].neurons[n].bias;
            }
        }
    }

    template<typename type>
    ANN<type>::~ANN(){
        layers.clear();
    }

    template<typename type>
    void ANN<type>::initialize(const std::vector<unsigned int>& layern, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights){
        assert(layern.size() >= 2);
        std::srand((clock()+time(NULL))/2);
        layers.resize(layern.size());
        for (unsigned int i = 0; i < layers.size(); i++){
            layers[i].init(layern[i], i, (layern.size()-1 == i) ? 0 : layern[i+1], this);
        }
        for (auto& r : ResWeights){
            AddResidualWeight(r.first, r.second, 1.0f);
        }
    }

    template<typename type>
    std::vector<type> ANN<type>::costvec(const std::vector<type>& target, unsigned int l){
        std::vector<type> out;
        for (unsigned int n = 0; n < layers[l].neurons.size(); n++){
            out.push_back(layers[l].neurons[n].activation - target[n]);
        }
        return out;
    }

    template<typename type>
    typename ANN<type>::NEURON& ANN<type>::IDtoN(NeuronID nID){
        return *&*(layers[nID.l].neurons.begin() + nID.n);
    }

    template<typename type>
    type ANN<type>::costavg(const std::vector<type>& target){
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
    std::vector<type> ANN<type>::forwardpropagate(const std::vector<type>& input){
        assert(input.size() == layers[0].neurons.size());
        for (auto& n : layers[0].neurons){
            n.activation = input[n.currentID] + n.bias;
        }
        for (unsigned int i = 1; i < layers.size() - 1; i++){
            layers[i].calcActs(*actfuncHID);
        }
        layers.back().calcActs(*actfuncOUT);
        std::vector<type> out;
        for (auto& n : layers.back().neurons){
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
        std::vector<unsigned int> layerinp;
        unsigned int ln = 0;
        unsigned int nn = 0;
        std::string tmp;
        std::getline(file, line);
        input.str(line);
        while (std::getline(input, tmp, ',')){
            layerinp.push_back(std::atoi((tmp.c_str())));
        }
        unsigned int l = 0, n = 0;
        initialize(layerinp, std::vector<std::pair<NeuronID, NeuronID>>{});
        while (std::getline(file, line)){
            std::stringstream().swap(input);
            input << line;
            std::getline(input, tmp, ',');
            l = atoi(tmp.c_str());
            std::getline(input, tmp, ',');
            n = atoi(tmp.c_str());
            for (auto& i : layers[l].neurons[n].outweights){
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
        for (unsigned int i = 0; i < layers.size(); i++){
            line += std::to_string(layers[i].neurons.size()) + ((i < layers.size() - 1) ? "," : "");
        }
        file << line.append("\n");
        for (unsigned int l = 0; l < layers.size(); l++){
            for (unsigned int n = 0; n < layers[l].neurons.size(); n++){
                line = std::to_string(l) + "," + std::to_string(n);
                if (l < layers.size() - 1){
                    for (unsigned int w = 0; w < layers[l].neurons[n].outweights.size(); w++){
                        line += "," + std::to_string(layers[l].neurons[n].outweights[w]);
                    }
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
        std::vector<unsigned int> layern(layerno);
        file.read(reinterpret_cast<char *>(layern.data()), layerno * sizeof(int));
        initialize(layern, std::vector<std::pair<NeuronID, NeuronID>>{});
        unsigned int layerid, neuronid, last = 0;
        while (file.read(reinterpret_cast<char *>(&layerid), sizeof(int))&&
            file.read(reinterpret_cast<char *>(&neuronid), sizeof(int))){
            if (last > layerid){
                break;
            }
            type weight = 0.0f;
            for (auto& w : layers[layerid].neurons[neuronid].outweights){
                file.read(reinterpret_cast<char *>(&weight), sizeof(type));
                w = weight;
            }
            type bias;
            file.read(reinterpret_cast<char *>(&bias), sizeof(type));
            layers[layerid].neurons[neuronid].bias = bias;
            last = layerid;
            unsigned int s_of_r;
            file.read(reinterpret_cast<char *>(&s_of_r), sizeof(int));
            if (s_of_r){
                layers[layerid].neurons[neuronid].resWeights.clear();
            }
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
        std::vector<unsigned int> layern;
        for (auto& l : layers){
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
    void ANN<type>::resetStructure(const std::vector<int>& layern, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights){
        initialize(layern, ResWeights);
    }

    template<typename type>
    void ANN<type>::backpropagate(const std::vector<type>& input, const std::vector<type>& target, type learn_rate, bool learn_rate_safety){
        assert(target.size() == layers.back().neurons.size());
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
                        res_w.push_back(std::pair<ResidualWeight& , type>(layers[l].neurons[i].resWeights[r], delta[i]));
                    }
                }
            }
            if(learn_rate_safety){
                jump_slowdown = 1/std::abs(costavg(target));
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
        if((layers.size()-1 > lID)){
            layers[lID].init(layers[lID].neurons.size(), lID, layers[lID+1].neurons.size(), this);  
            for(unsigned int l = lID; l < layers.size(); l++){
                for(unsigned int n = 0; n < layers[l].neurons.size(); n++){
                    for(unsigned int r = 0; r < layers[l].neurons[n].resWeights.size(); r++){
                        unsigned int from_n = layers[l].neurons[n].resWeights[r].from.n;
                        if(from_n >= n){
                            layers[l].neurons[n].resWeights.erase(layers[l].neurons[n].resWeights.begin()+r);
                            r--;
                        }
                    }
                }
            }      
        }
        else{
            layers[lID].init(layers[lID].neurons.size(), lID, 0, this);            
        }
        if(lID > 0){
            layers[lID-1].init(layers[lID-1].neurons.size(), lID-1, layers[lID].neurons.size(), this);
        }
    }

    template<typename type>
    void ANN<type>::addNeuron(unsigned int lID){
        assert(lID >= 0 && lID < layers.size());
        layers[lID].neurons.push_back(NEURON());
        if((layers.size()-1 > lID)){
            layers[lID].init(layers[lID].neurons.size(), lID, layers[lID+1].neurons.size(), this);        
        }
        else{
            layers[lID].init(layers[lID].neurons.size(), lID, 0, this);            
        }
        if(lID > 0){
            layers[lID-1].init(layers[lID-1].neurons.size(), lID-1, layers[lID].neurons.size(), this);
        }
        if(lID < layers.size()-1){ 
            layers[lID].neurons.back().initializeweights();
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
            for(unsigned int l = lID; l < layers.size(); l++){
                for(unsigned int n = 0; n < layers[l].neurons.size(); n++){
                    for(unsigned int r = 0; r < layers[l].neurons[n].resWeights.size(); r++){
                        unsigned int from_l = layers[l].neurons[n].resWeights[r].from.l;
                        if(from_l >= l){
                            layers[l].neurons[n].resWeights.erase(layers[l].neurons[n].resWeights.begin()+r);
                            r--;
                        }
                    }
                }
            }
        }
        else if(lID == 0){
            layers.erase(layers.begin());
            for(auto& l : layers){
                l.curr_l--;
            }
            for(unsigned int l = 0; l < layers.size(); l++){
                for(unsigned int n = 0; n < layers[l].neurons.size(); n++){
                    for(unsigned int r = 0; r < layers[l].neurons[n].resWeights.size(); r++){
                        unsigned int from_l = layers[l].neurons[n].resWeights[r].from.l;
                        if(from_l >= l){
                            layers[l].neurons[n].resWeights.erase(layers[l].neurons[n].resWeights.begin()+r);
                            r--;
                        }
                    }
                }
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
        layers.insert(lID+layers.begin(), LAYER());
        if(lID > 0 && lID < layers.size()-1){
            layers[lID].init(s, lID, layers[lID+1].neurons.size(), this);
            for(unsigned int n = 0; n < layers[lID].neurons.size(); n++){
                layers[lID].neurons[n].initializeweights();
                if(layers[lID-1].neurons.size() > n){
                    for(unsigned int o = 0; o < layers[lID-1].neurons[n].outweights.size(); o++){
                        layers[lID].neurons[n].outweights[o] = layers[lID-1].neurons[n].outweights[o];
                    }
                }
            }
            layers[lID-1].init(layers[lID-1].neurons.size(), lID-1, layers[lID].neurons.size(), this);
            for(unsigned int l = lID+1; l < layers.size(); l++){
                layers[l].curr_l++;
            }
        }
        else if(lID == 0){
            layers[lID].init(s, lID, layers[lID+1].neurons.size(), this);
            for(unsigned int l = 0; l < layers.size(); l++){
                layers[l].curr_l++;
            }
        }
        else{
            layers[layers.size()-1].init(s, lID, 0, this);
            layers[lID-1].init(layers[lID-1].neurons.size(), lID-1, layers[lID].neurons.size(), this);
        }
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
    void ANN<type>::AddResidualWeight(NeuronID from, NeuronID to, type weight){
        assert(from.l < layers.size() || to.l < layers.size());
        assert(from.n < layers[from.l].neurons.size() || to.n < layers[to.l].neurons.size());
        assert(from.l < to.l);
        layers[to.l].neurons[to.n].resWeights.push_back(ResidualWeight(from, weight));
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
        resWeights.clear();
    }

    template<typename type> 
    void ANN<type>::NEURON::initializeweights(){
        for (type& i : outweights)
            i = 2*rand()/(type)RAND_MAX;
        bias = 2*rand()/(type)RAND_MAX;
    }

    template<typename type> 
    struct ANN<type>::LAYER{
        public:

            ANN<type>* belong_to = nullptr;
            std::vector<NEURON> neurons;
            unsigned int curr_l = 0;
            LAYER(unsigned int n, unsigned int curr, unsigned int next_s, ANN<type>* pbelong_to);
            LAYER() = default;
            ~LAYER();
            void init(unsigned int n, unsigned int curr, unsigned int next_s, ANN<type>* pbelong_to);
            void calcActs(type(actfunc)(type));

    };

    template<typename type>
    ANN<type>::LAYER::LAYER(unsigned int n, unsigned int curr, unsigned int next_s, ANN<type>* pbelong_to){
        init(n, curr, next_s, pbelong_to);
        belong_to = pbelong_to;
    }

    template<typename type>
    ANN<type>::LAYER::~LAYER(){
        neurons.clear();
        belong_to = nullptr;
    }

    template<typename type>
    void ANN<type>::LAYER::init(unsigned int n, unsigned int curr, unsigned int next_s, ANN<type>* pbelong_to){
        assert(n > 0);
        neurons.resize(n, NEURON());
        curr_l = curr;
        unsigned int x = 0;
        belong_to = pbelong_to;
        for (auto& i : neurons){
            i.currentID = x;
            x++;
            i.outweights.resize(next_s);
            i.initializeweights();   
        }
    }

    template<typename type>
    void ANN<type>::LAYER::calcActs(type(actfunc)(type)){
        for(auto& n : belong_to->layers[curr_l].neurons){
            n.activation = 0.0f;
            for(auto& np : belong_to->layers[curr_l-1].neurons){
                n.activation += np.outweights[n.currentID]*np.activation;
            }
            for(auto& r : n.resWeights){
                n.activation += belong_to->IDtoN(r.from).activation * r.weight;
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
    class EVO_TRAINER{

        private:

            std::vector<ANN<type>*> networks;
            type constant = 0.0f;
            bool diff = false;
            void mutate(const ANN<type>& net, type mutate_rate);
            type (*fitness)(ANN<type>& net);
        public:

            EVO_TRAINER() = default;
            EVO_TRAINER(const unsigned int n_of_networks, const std::vector<unsigned int>& structure, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights, type(*actHID)(type), type(*actOUT)(type), type (*fitnessp)(ANN<type>& net), type re_structure_constant);
            EVO_TRAINER(const unsigned int n_of_networks, const ANN<type>& template_ANN, type (*fitnessp)(ANN<type>& net), type mutate_rate, type re_structure_constant);
            ~EVO_TRAINER();
            void mutate_generation(bool re_structure, type mutate_rate);
            void mutate_generation(bool re_structure, ANN<type>& net, type mutate_rate);
            ANN<type>& best_speciman();
            void re_init(const unsigned int n_of_networks, const std::vector<unsigned int>& structure, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights, type(*actHID)(type), type(*actOUT)(type), type (*fitnessp)(ANN<type>& net), type re_structure_constant);
            void re_init(const unsigned int n_of_networks, const ANN<type>& template_ANN, type (*fitnessp)(ANN<type>& net), type mutate_rate, type re_structure_constant);

    };

    template<typename type>
    EVO_TRAINER<type>::EVO_TRAINER(const unsigned int n_of_networks, const std::vector<unsigned int>& structure, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights, type(*actHID)(type), type(*actOUT)(type), type (*fitnessp)(ANN<type>& net), type re_structure_constant){
        std::srand((clock()+time(NULL))/2);
        for(unsigned int nn = 0; nn < n_of_networks; nn++){
            networks.push_back(new ANN<type>(structure, ResWeights, actHID, actOUT));
        }
        fitness = fitnessp;
        constant = re_structure_constant;
    }

    template<typename type>
    EVO_TRAINER<type>::EVO_TRAINER(const unsigned int n_of_networks, const ANN<type>& template_ANN, type (*fitnessp)(ANN<type>& net), type mutate_rate, type re_structure_constant){
        for(unsigned int nn = 0; nn < n_of_networks; nn++){
            networks.push_back(new ANN(template_ANN));
        }
        fitness = fitnessp;
        mutate(template_ANN, mutate_rate);
        constant = re_structure_constant;
    }

    template<typename type>
    EVO_TRAINER<type>::~EVO_TRAINER(){
        for (auto& network : networks) {
            delete network;
        }
        networks.clear();
    }

    template<typename type>
    void EVO_TRAINER<type>::mutate(const ANN<type>& net, type mutate_rate){
        for(auto& nn : networks){
            for(unsigned int l = 0; l < nn->layers.size() && l < net.layers.size(); l++){
                for(unsigned int n = 0; n < nn->layers[l].neurons.size() && n < net.layers[l].neurons.size(); n++){
                    for(unsigned int w = 0; w < nn->layers[l].neurons[n].outweights.size() && w < net.layers[l].neurons[n].outweights.size(); w++){
                        nn->layers[l].neurons[n].outweights[w] = (net.layers[l].neurons[n].outweights[w] + mutate_rate*(2*rand()/(type)RAND_MAX));
                    }
                    for(unsigned int rw = 0; rw < nn->layers[l].neurons[n].resWeights.size() && rw < net.layers[l].neurons[n].resWeights.size(); rw++){
                        nn->layers[l].neurons[n].resWeights[rw].weight = (nn->layers[l].neurons[n].resWeights[rw].weight + mutate_rate*(2*rand()/(type)RAND_MAX));
                    }
                    nn->layers[l].neurons[n].bias = net.layers[l].neurons[n].bias + mutate_rate*(2*rand()/(type)RAND_MAX);
                }
            }
        }
    }

    template<typename type>
    void EVO_TRAINER<type>::mutate_generation(bool re_structure, type mutate_rate){
        ANN<type>* best_nn = &best_speciman();
        if(re_structure || diff){
            for(unsigned int nn = 0; nn < networks.size(); nn++){
                if(best_nn != networks[nn]){
                    delete networks[nn];
                    networks[nn] = new ANN(*best_nn);
                }
            }
            diff = false;
        }
        mutate(*best_nn, mutate_rate);
        if(re_structure){
            for(unsigned int nn = 0; nn < networks.size(); nn++){
                bool to_add_l = int(mutate_rate * double(rand() * constant))%2;
                if(to_add_l){
                    networks[nn]->addLayer(networks[nn]->layers.size()/2, 5);
                }
                else if(networks[nn]->layers.size() != 2){
                    networks[nn]->deleteLayer(networks[nn]->layers.size()/2);
                }
                for(unsigned int l = 1; l < networks[nn]->layers.size()-1; l++){
                    bool to_add_n = int(mutate_rate * double(rand() * constant))%2;
                    if(to_add_n){
                        networks[nn]->addNeuron(l);
                    }
                    else if (networks[nn]->layers[l].neurons.size() > 1){
                        networks[nn]->deleteNeuron(l);
                    }
                    for(unsigned int n = 0; n < networks[nn]->layers[l].neurons.size(); n++){
                        bool to_add_r = int(mutate_rate * double(rand() * constant))%2;
                        if(to_add_r){
                            unsigned int from_l = int(l*(double(rand())/RAND_MAX));
                            unsigned int from_n = int(networks[nn]->layers[from_l].neurons.size()*(double(rand())/RAND_MAX));
                            type weight = 2*rand()/(type)RAND_MAX;
                            networks[nn]->AddResidualWeight(NeuronID(from_l, from_n), NeuronID(l, n), weight);
                        }
                        else if(networks[nn]->layers[l].neurons[n].resWeights.size() > 0){
                            unsigned int from_l = int(l*(double(rand())/RAND_MAX));
                            unsigned int from_n = int(networks[nn]->layers[from_l].neurons.size()*(double(rand())/RAND_MAX));
                            unsigned int to_delete = rand()%networks[nn]->layers[l].neurons[n].resWeights.size();
                            networks[nn]->DeleteResidualWeight(NeuronID(from_l, from_n), NeuronID(l, n));
                        }
                    }
                }
            }
            diff = true;   
        }
    }

    template<typename type>
    void EVO_TRAINER<type>::mutate_generation(bool re_structure, ANN<type>& net, type mutate_rate){
        if(re_structure || diff){
            for(unsigned int nn = 0; nn < networks.size(); nn++){
                if(best_nn != networks[nn]){
                    delete networks[nn];
                    networks[nn] = new ANN(net);
                }
            }
            diff = false;
        }
        mutate(net, mutate_rate);
        if(re_structure){
            for(unsigned int nn = 0; nn < networks.size(); nn++){
                bool to_add_l = int(mutate_rate * double(rand() * constant))%2;
                if(to_add_l){
                    networks[nn]->addLayer(networks[nn]->layers.size()/2, 5);
                }
                else if(networks[nn]->layers.size() != 2){
                    networks[nn]->deleteLayer(networks[nn]->layers.size()/2);
                }
                for(unsigned int l = 1; l < networks[nn]->layers.size()-1; l++){
                    bool to_add_n = int(mutate_rate * double(rand() * constant))%2;
                    if(to_add_n){
                        networks[nn]->addNeuron(l);
                    }
                    else if (networks[nn]->layers[l].neurons.size() > 1){
                        networks[nn]->deleteNeuron(l);
                    }
                    for(unsigned int n = 0; n < networks[nn]->layers[l].neurons.size(); n++){
                        bool to_add_r = int(mutate_rate * double(rand() * constant))%2;
                        if(to_add_r){
                            unsigned int from_l = int(l*(double(rand())/RAND_MAX));
                            unsigned int from_n = int(networks[nn]->layers[from_l].neurons.size()*(double(rand())/RAND_MAX));
                            type weight = 2*rand()/(type)RAND_MAX;
                            networks[nn]->AddResidualWeight(NeuronID(from_l, from_n), NeuronID(l, n), weight);
                        }
                        else if(networks[nn]->layers[l].neurons[n].resWeights.size() > 0){
                            unsigned int from_l = int(l*(double(rand())/RAND_MAX));
                            unsigned int from_n = int(networks[nn]->layers[from_l].neurons.size()*(double(rand())/RAND_MAX));
                            unsigned int to_delete = rand()%networks[nn]->layers[l].neurons[n].resWeights.size();
                            networks[nn]->DeleteResidualWeight(NeuronID(from_l, from_n), NeuronID(l, n));
                        }
                    }
                }
            }
            diff = true;   
        }        
    }

    template<typename type>
    ANN<type>& EVO_TRAINER<type>::best_speciman(){
        type best_ff = fitness(*networks[0]);
        ANN<type>* best_nn = networks[0];
        for(unsigned int nn = 1; nn < networks.size(); nn++){
            type current_ff = fitness(*networks[nn]);
            if(current_ff > best_ff){
                best_ff = current_ff;
                best_nn = networks[nn];
            }
        }
        return *best_nn;
    }

    template<typename type>
    void EVO_TRAINER<type>::re_init(const unsigned int n_of_networks, const std::vector<unsigned int>& structure, const std::vector<std::pair<NeuronID, NeuronID>>& ResWeights, type(*actHID)(type), type(*actOUT)(type), type (*fitnessp)(ANN<type>& net), type re_structure_constant){
        for (auto& network : networks) {
            delete network;
        }
        networks.clear();      
        std::srand((clock()+time(NULL))/2);
        for(unsigned int nn = 0; nn < n_of_networks; nn++){
            networks.push_back(new ANN<type>(structure, ResWeights, actHID, actOUT));
        }
        fitness = fitnessp;
        constant = re_structure_constant;        
    }

    template<typename type>
    void EVO_TRAINER<type>::re_init(const unsigned int n_of_networks, const ANN<type>& template_ANN, type (*fitnessp)(ANN<type>& net), type mutate_rate, type re_structure_constant){
        for (auto& network : networks) {
            delete network;
        }
        networks.clear();
        for(unsigned int nn = 0; nn < n_of_networks; nn++){
            networks.push_back(new ANN(template_ANN));
        }
        fitness = fitnessp;
        mutate(template_ANN, mutate_rate);
        constant = re_structure_constant;          
    }

#endif