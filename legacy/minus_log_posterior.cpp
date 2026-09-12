#include <Eigen/Dense>
#include <cmath>
#include <tuple>

Eigen::MatrixXd fourier_components(const Eigen::VectorXd& t_days, double period, int n) {
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 1, n) * (2 * M_PI / period);
    Eigen::MatrixXd angles = t_days * x.transpose();
    
    Eigen::MatrixXd result(t_days.size(), 2 * n);
    result.leftCols(n) = angles.array().cos();
    result.rightCols(n) = angles.array().sin();
    
    return result;
}

std::tuple<double, double, Eigen::VectorXd, Eigen::VectorXd> extract_params(const Eigen::VectorXd& params) {
    double k = params(0);
    double m = params(1);
    Eigen::VectorXd delta = params.segment(2, 25);
    Eigen::VectorXd beta = params.tail(params.size() - 27);
    return std::make_tuple(k, m, delta, beta);
}

extern "C" {
    double minus_log_posterior(const double* params,
                               int params_size,
                               const double* t_scaled,
                               int t_scaled_size,
                               const double* change_points,
                               int change_points_size,
                               double scale_period,
                               const double* normalized_y,
                               int normalized_y_size,
                               double sigma_obs,
                               double sigma_k,
                               double sigma_m,
                               double sigma,
                               double tau);
}

double minus_log_posterior(const double* params,
                           int params_size,
                           const double* t_scaled,
                           int t_scaled_size,
                           const double* change_points,
                           int change_points_size,
                           double scale_period,
                           const double* normalized_y,
                           int normalized_y_size,
                           double sigma_obs,
                           double sigma_k,
                           double sigma_m,
                           double sigma,
                           double tau) {
    Eigen::Map<const Eigen::VectorXd> params_vec(params, params_size);
    Eigen::Map<const Eigen::VectorXd> t_scaled_vec(t_scaled, t_scaled_size);
    Eigen::Map<const Eigen::VectorXd> change_points_vec(change_points, change_points_size);
    Eigen::Map<const Eigen::VectorXd> normalized_y_vec(normalized_y, normalized_y_size);

    double k, m;
    Eigen::VectorXd delta, beta;
    std::tie(k, m, delta, beta) = extract_params(params_vec);

    // Trend component
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(t_scaled_vec.size());


    Eigen::MatrixXd A = (t_scaled_vec.replicate(1, change_points_vec.size()).array() > change_points_vec.transpose().replicate(t_scaled_vec.size(), 1).array()).cast<double>();
    Eigen::VectorXd gamma = -delta.array() * change_points_vec.array();
    Eigen::VectorXd g = (k * ones + A * delta).array() * t_scaled_vec.array() + (m * ones + A * gamma).array();

    // Seasonality component
    double period = 365.25 / scale_period;
    Eigen::MatrixXd x = fourier_components(t_scaled_vec, period, 10);
    Eigen::VectorXd s = x * beta;

    Eigen::VectorXd y_pred = g + s;

    double sum_squared_diff = (normalized_y_vec - y_pred).array().square().sum();

    double minus_log_posterior_value = sum_squared_diff / (2 * std::pow(sigma_obs, 2)) + 
                                       std::pow(k, 2) / (2 * std::pow(sigma_k, 2)) + 
                                       std::pow(m, 2) / (2 * std::pow(sigma_m, 2)) + 
                                       beta.array().square().sum() / (2 * std::pow(sigma, 2)) + 
                                       delta.array().abs().sum() / tau;

    return minus_log_posterior_value;
}

// To compile, on Mac OS, run the following command in the terminal:
// g++ -std=c++17 -shared -fPIC -Ofast -o libminus_log_posterior_5.so minus_log_posterior_5.cpp -I/opt/homebrew/opt/eigen/include/eigen3