#pragma once

#include <cmath>

double logistic(double x) {
	return 1.0 / (1 + std::exp(-x));
}

double logisticDer(double x) {
	return logistic(x) * (1 - logistic(x));
}