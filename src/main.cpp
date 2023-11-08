#include "../include/ANN.h"
#include <cmath>
#include <iostream>
#include <vector>
//#include "../include/ANN_TRAINER.h"

std::vector<int> layers = {2, 1};
std::vector<double> input(layers[0], 1.0f);
std::vector<double> target(layers.back(), 0.0f);
std::vector<std::pair<NeuronID, NeuronID>> rw = {std::pair<NeuronID, NeuronID>(NeuronID(0, 0), NeuronID(1, 0))};

double foo(ANN<double>& net)
{
    net.forwardpropagate(input);
    return net.costavg(target);
}

int main(/*int argv, char* argc[]*/)
{

    int epochs = /*std::atoi(argc[1])*/ 10;
    float learn_or_mutate_rate = /*std::atof(argc[2])*/ 0.1f;
    ANN<double> nets(layers,rw,std::tanh,std::tanh);

    std::cin.get();
    return 0;
}
