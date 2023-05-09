#pragma once

#include <cmath>

double MSE(double y, double target) noexcept {
	return std::pow(target - y, 2);
}

double MSEDeriv(double y, double target) noexcept {
	return 2 * (target - y);
}