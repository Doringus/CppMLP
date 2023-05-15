#include "mlp.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>

#include "errorfuncs.hpp"


MLP::MLP(const std::vector<layerConfig_t>& hiddenLayers, const ErrorFunc& errorFunc, const ErrorFunc& errorFuncDer)
		: m_ErrorFunc(errorFunc), m_ErrorFuncDer(errorFuncDer) {
	m_Layers.reserve(hiddenLayers.size());
	m_Layers.emplace_back(hiddenLayers.at(0).neuronCount, hiddenLayers.at(0).activationFunc, hiddenLayers.at(0).activationFuncDer);
	for (int i = 1; i < hiddenLayers.size(); ++i) {
		m_Layers.emplace_back(&m_Generator, hiddenLayers[i].neuronCount, hiddenLayers[i - 1].neuronCount,
			hiddenLayers[i].activationFunc, hiddenLayers[i].activationFuncDer);
	}
}


void MLP::fit(std::vector<TrainingElement>& inputData, size_t epochCount) {
	m_Layers[0].createWeights(inputData[0].first.size(), &m_Generator);

	std::vector<double> targets;
	for (auto& target : inputData | std::views::values) {
		targets.push_back(target);
	}


	for (int i = 0; i < epochCount; ++i) {
		std::vector<double> predicted;
		for (auto& [data, target] : inputData) {
			std::vector<double> layerInput = data;

			for(auto& layer : m_Layers) {
				layerInput = layer.forwardPass(layerInput);
			}

			std::vector<double> currentErrors;
			/// Calculate loss func derivative
			predicted.push_back(layerInput[0]);
			for (double output : layerInput) {
				double dx = m_ErrorFuncDer(output, target);
				currentErrors.push_back(dx);
			}

			for (auto& m_Layer : std::ranges::reverse_view(m_Layers)) {
				currentErrors = m_Layer.backwardPass(currentErrors);
			}


		}
		double mse = MSE(targets, predicted);
		std::cout << "MSE = " << mse << "\n";
	}

}

std::vector<double> MLP::predict(const std::vector<double>& data) {
	std::vector<double> layerInput = data;

	for (auto& layer : m_Layers) {
		layerInput = layer.forwardPass(layerInput);
	}

	return layerInput;
}

