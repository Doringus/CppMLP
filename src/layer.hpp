#pragma once

#include <functional>
#include <random>

using ActivationFunc = std::function<double(double)>;

class Layer {
public:
	Layer(size_t neuronCount, const ActivationFunc& activationFunc, const ActivationFunc& activationFuncDerivative) noexcept;
	Layer(std::mt19937* generator, size_t neuronCount, size_t inputsPerNeuron, const ActivationFunc& activationFunc, const ActivationFunc& activationFuncDerivative);

	void createWeights(size_t inputsPerNeuron, std::mt19937* generator);
	std::vector<double> forwardPass(const std::vector<double>& input);
	std::vector<double> backwardPass(const std::vector<double>& errorInput);
private:
	ActivationFunc m_ActivationFunc, m_ActivationFuncDerivative;
	size_t m_NeuronCount;
	size_t m_InputsPerNeuron;
	std::vector<double> m_ActivationFuncInput;
	std::vector<double> m_Input;
	std::vector<double> m_Weights;
};