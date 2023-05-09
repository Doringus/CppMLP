#pragma once

#include <functional>
#include <random>

using ActivationFunc = std::function<double(double)>;
using ErrorFunc = std::function<double(double, double)>;
using TrainingElement = std::pair<std::vector<double>, double>;



struct layerConfig_t {
	size_t neuronCount;
	ActivationFunc activationFunc;
	ActivationFunc activationFuncDer;
};

class MLP {

	struct layer_t {
		layer_t(std::mt19937* generator, size_t inputDataSize, size_t neuronCount,
			const ActivationFunc& activationFunc, const ActivationFunc& activationFuncDer);
		layer_t(size_t neuronCount, const ActivationFunc& activationFunc, const ActivationFunc& activationFuncDer);

		void createWeights(size_t weightsCount, std::mt19937* generator);

		/**
		 * @details input weights for all neurons
		 */
		std::vector<double> weights;

		/**
		 * @details outputs from all neurons
		 */
		std::vector<double> outputs;

		/**
		 * @details - result of sum(w*x + b) for all neurons
		 */
		std::vector<double> activationInput;

		size_t inputCountPerNeuron;
		ActivationFunc activationFunc;
		ActivationFunc activationFuncDer;
	};

public:
	MLP(const std::vector<layerConfig_t>& hiddenLayers, const ErrorFunc& errorFunc, const ErrorFunc& errorFuncDer);

	void fit(std::vector<TrainingElement>& inputData, size_t epochCount);
private:
	void forward(const std::vector<double>& data);
	void backprop(double target, const std::vector<double>& inputData);
	std::vector<double> backpropLayer(const std::vector<double>& layerInput, const std::vector<double>& errorsInput, layer_t& layer);
private:
	std::vector<layer_t> m_Layers;
	ErrorFunc m_ErrorFunc, m_ErrorFuncDer;
	std::mt19937 m_Generator;
};
