#pragma once

#include <functional>
#include <random>

#include "layer.hpp"

using ActivationFunc = std::function<double(double)>;
using ErrorFunc = std::function<double(double, double)>;
using TrainingElement = std::pair<std::vector<double>, double>;

struct layerConfig_t {
	size_t neuronCount;
	ActivationFunc activationFunc;
	ActivationFunc activationFuncDer;
};


class MLP {
public:
	MLP(const std::vector<layerConfig_t>& hiddenLayers, const ErrorFunc& errorFunc, const ErrorFunc& errorFuncDer);

	void fit(std::vector<TrainingElement>& inputData, size_t epochCount);
	std::vector<double> predict(const std::vector<double>& data);
private:
	//std::vector<layer_t> m_Layers;
	ErrorFunc m_ErrorFunc, m_ErrorFuncDer;
	std::mt19937 m_Generator;
	std::vector<Layer> m_Layers;
};
