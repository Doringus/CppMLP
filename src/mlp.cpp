#include "mlp.hpp"

#include <algorithm>
#include <numeric>
#include <ranges>

MLP::MLP(const std::vector<layerConfig_t>& hiddenLayers, const ErrorFunc& errorFunc, const ErrorFunc& errorFuncDer)
		: m_Generator((std::random_device())()), m_ErrorFunc(errorFunc), m_ErrorFuncDer(errorFuncDer) {
	m_Layers.reserve(hiddenLayers.size());
	m_Layers.emplace_back(hiddenLayers.at(0).neuronCount, hiddenLayers.at(0).activationFunc, hiddenLayers.at(0).activationFuncDer);
	for(int i = 1; i < hiddenLayers.size(); ++i) {
		m_Layers.emplace_back(&m_Generator, m_Layers[i - 1].outputs.size() * hiddenLayers[i].neuronCount, 
			hiddenLayers[i].neuronCount, hiddenLayers[i].activationFunc, hiddenLayers[i].activationFuncDer);
	}
	
}

void MLP::fit(std::vector<TrainingElement>& inputData, size_t epochCount) {
	m_Layers[0].createWeights(inputData[0].first.size() * m_Layers[0].outputs.size(), &m_Generator);

	for(int i = 0; i < epochCount; ++i) {
		for(auto& [data, target] : inputData) {
			forward(data);
			backprop(target, data);
		}
	}

}

void MLP::forward(const std::vector<double>& data) {
	auto inputData = std::ref(data);
	std::ranges::for_each(m_Layers, [inputData](layer_t& layer) mutable {

		for(size_t currentNeuron = 0; currentNeuron < layer.outputs.size(); ++currentNeuron) {
			auto neuronWeightsBegin = layer.weights.begin();
			auto neuronWeightsEnd = layer.weights.begin();
			std::advance(neuronWeightsBegin, currentNeuron * layer.inputCountPerNeuron);
			std::advance(neuronWeightsEnd, currentNeuron * layer.inputCountPerNeuron + layer.inputCountPerNeuron);

			const double activationFuncInput = std::inner_product(neuronWeightsBegin, neuronWeightsEnd, inputData.get().begin(), 0.0);
			layer.activationInput[currentNeuron] = activationFuncInput;
			layer.outputs[currentNeuron] = std::invoke(layer.activationFunc, activationFuncInput);

		}
		inputData = std::ref(layer.outputs);
	});
}

void MLP::backprop(double target, const std::vector<double>& inputData) {
	std::vector<double> currentErrors;
	/// Calculate mse errors for last layer
	for (double output : m_Layers.back().outputs) {
		double dx = m_ErrorFuncDer(output, target);
		currentErrors.push_back(dx);
	}

	for(size_t layerIndex = m_Layers.size() - 1; layerIndex > 0; --layerIndex) {
		currentErrors = backpropLayer(m_Layers[layerIndex - 1].outputs, currentErrors, m_Layers[layerIndex]);
	}
	backpropLayer(inputData, currentErrors, m_Layers[0]);
}

std::vector<double> MLP::backpropLayer(const std::vector<double>& layerInput, const std::vector<double>& errorsInput, layer_t& layer) {
	std::vector<double> errorsToPropagate(layerInput.size(), 0.0);

	for (size_t neuronIndex = 0; neuronIndex < layer.outputs.size(); ++neuronIndex) {
		const double dx = errorsInput[neuronIndex];
		const double db = dx * layer.activationFuncDer(layer.activationInput[neuronIndex]);
		/// find weights delta`s for current neuron
		for (size_t inputDataIndex = 0; inputDataIndex < layerInput.size(); ++inputDataIndex) {
			double delta = layerInput[inputDataIndex] * db; /// TODO: add alpha
			/// send error to previous layer
			errorsToPropagate[inputDataIndex] += db * layer.weights[neuronIndex * layer.inputCountPerNeuron + inputDataIndex];
			/// change weights
			layer.weights[neuronIndex * layer.inputCountPerNeuron + inputDataIndex] -= delta;
		}
	}

	return errorsToPropagate;
}

MLP::layer_t::layer_t(std::mt19937* generator, size_t inputDataSize, size_t neuronCount, 
                      const ActivationFunc& activationFunc, const ActivationFunc& activationFuncDer)
	: outputs(neuronCount, 0.0), activationInput(neuronCount, 0.0),
	  inputCountPerNeuron(inputDataSize / neuronCount), activationFunc(activationFunc), activationFuncDer(activationFuncDer) {
	createWeights(inputDataSize, generator);
}

MLP::layer_t::layer_t(size_t neuronCount, const ActivationFunc& activationFunc, const ActivationFunc& activationFuncDer)
			: layer_t(nullptr, 0, neuronCount, activationFunc, activationFuncDer) { }

void MLP::layer_t::createWeights(size_t weightsCount, std::mt19937* generator) {
	inputCountPerNeuron = weightsCount / outputs.size();
	weights = std::vector(weightsCount, 0.0);
	std::uniform_real_distribution<> dist(-1.0, 1.0);
	std::ranges::generate(weights, [&dist, generator]() {
		return dist(*generator);
	});
}