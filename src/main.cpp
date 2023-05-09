#include <iostream>

#include "mlp.hpp"
#include "activationfuncs.hpp"
#include "errorfuncs.hpp"

int main() {
	std::vector<layerConfig_t> layers = {
		layerConfig_t{5, logistic, logisticDer},
		layerConfig_t{5, logistic, logisticDer},
		layerConfig_t{5, logistic, logisticDer},
		layerConfig_t{1, logistic, logisticDer}
	};
	MLP mlp(layers, MSE, MSEDeriv);
	std::vector<TrainingElement> data = {
		{{ 0.5, 1.2, 0.8, 3.2 }, 1.0},
		{{ 0.5, 1.2, 0.8, 3.2 }, 1.0},
		{{ 0.5, 1.2, 0.8, 3.2 }, 1.0},
		{{ 0.5, 1.2, 0.8, 3.2 }, 1.0},
	};
	mlp.fit(data, 10);
	return 0;
}