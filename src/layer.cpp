#include "layer.hpp"

#include <numeric>

Layer::Layer(size_t neuronCount, const ActivationFunc& activationFunc, const ActivationFunc& activationFuncDerivative) noexcept
		: Layer(nullptr, neuronCount, 0, activationFunc, activationFuncDerivative) { }


Layer::Layer(std::mt19937* generator, size_t neuronCount, size_t inputsPerNeuron, const ActivationFunc& activationFunc,
             const ActivationFunc& activationFuncDerivative) : m_ActivationFunc(activationFunc),
             m_ActivationFuncDerivative(activationFuncDerivative), m_NeuronCount(neuronCount), m_InputsPerNeuron(inputsPerNeuron),
			 m_ActivationFuncInput(neuronCount, 0.0)
{
	createWeights(inputsPerNeuron, generator);
}

void Layer::createWeights(size_t inputsPerNeuron, std::mt19937* generator) {
	m_InputsPerNeuron = inputsPerNeuron;
	m_Weights = std::vector(m_NeuronCount * inputsPerNeuron, 0.0);
	std::uniform_real_distribution<> dist(-1.0, 1.0);
	std::ranges::generate(m_Weights, [&dist, generator]() {
		return dist(*generator);
	});
}

std::vector<double> Layer::forwardPass(const std::vector<double>& input) {
	m_Input = input;
	std::vector<double> result;
	result.reserve(m_NeuronCount);
	for (size_t currentNeuron = 0; currentNeuron < m_NeuronCount; ++currentNeuron) {
		auto neuronWeightsBegin = m_Weights.begin();
		auto neuronWeightsEnd = m_Weights.begin();
		std::advance(neuronWeightsBegin, currentNeuron * m_InputsPerNeuron);
		std::advance(neuronWeightsEnd, currentNeuron * m_InputsPerNeuron + m_InputsPerNeuron);

		const double activationFuncInput = std::inner_product(neuronWeightsBegin, neuronWeightsEnd, input.begin(), 0.0);
		m_ActivationFuncInput[currentNeuron] = activationFuncInput;
		result.push_back(m_ActivationFunc(activationFuncInput));
	}
	return result; 
}

std::vector<double> Layer::backwardPass(const std::vector<double>& errorInput) {
	std::vector<double> errorsToPropagate(m_InputsPerNeuron, 0.0);
	for(size_t neuronIndex = 0; neuronIndex < errorInput.size(); ++neuronIndex) {
		double dx = errorInput[neuronIndex];
		double db = dx * m_ActivationFuncDerivative(m_ActivationFuncInput[neuronIndex]) * 0.005;

		for(size_t weightIndex = 0; weightIndex < m_InputsPerNeuron; weightIndex++) {
			double delta = db * m_Input[weightIndex];
			double err = db * m_Weights[neuronIndex * m_InputsPerNeuron + weightIndex];
			errorsToPropagate[weightIndex] += err;
			m_Weights[neuronIndex * m_InputsPerNeuron + weightIndex] += delta;
		}
	}
	return errorsToPropagate;
}

