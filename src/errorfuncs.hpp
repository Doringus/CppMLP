#pragma once

#include <cmath>
#include <algorithm>
#include <cassert>
#include <numeric>

static double MSELoss(double y, double target) noexcept {
	return std::pow(target - y, 2);
}

static double MSEDeriv(double y, double target) noexcept {
	return 2 * (target - y);
}

static double CELoss(double y, double target) noexcept {
	return -y * std::log(target);
}

static double CEDeriv(double y, double target) noexcept {
	return (y - CELoss(y, target)) / (CELoss(y, target) * (1 - CELoss(y, target)));
}

static double MSE(const std::vector<double>& expected, const std::vector<double>& predicted) {
	assert(expected.size() == predicted.size());
	double error = 0;
	for(int i = 0; i < expected.size(); ++i) {
		error += std::pow(predicted[i] - expected[i], 2);
	}
	return error / expected.size();
}

static double MAE(const std::vector<double>& expected, const std::vector<double>& predicted) {
	assert(expected.size() == predicted.size());
	double error = 0;
	for (int i = 0; i < expected.size(); ++i) {
		error += std::abs(predicted[i] - expected[i]);
	}
	return error / expected.size();
}

static double MAPE(const std::vector<double>& expected, const std::vector<double>& predicted) {
	assert(expected.size() == predicted.size());
	double error = 0;
	for (int i = 0; i < expected.size(); ++i) {
		error += std::abs(predicted[i] - expected[i]) / std::abs(std::max(0.00001, expected[i]));
	}
	return error / expected.size();
}

static double rsquare(const std::vector<double>& expected, const std::vector<double>& predicted) {
	assert(expected.size() == predicted.size());
	double expectedMean = std::accumulate(expected.begin(), expected.end(), 0.0) / static_cast<double>(expected.size());
	double error = 0;
	for (int i = 0; i < expected.size(); ++i) {
		error += std::pow(predicted[i] - expected[i], 2);
	}
	error /= expected.size();
	double totalSum = 0.0;
	for(int i = 0; i < predicted.size(); ++i) {
		totalSum += std::pow(expected[i] - expectedMean, 2);
	}
	return 1 - error / totalSum;
}