#pragma once

#include <cmath>

double logistic(double x) {
	return 1.0 / (1 + std::exp(-x));
}

double logisticDer(double x) {
	return logistic(x) * (1 - logistic(x));
}

double relu(double x) {
	return std::max(0.0, x);
}

double reluDer(double x) {
	return x > 0 ? 1 : 0;
}

double linear(double x) {
	return x;
}

double linearDer(double x) {
	return 1;
}