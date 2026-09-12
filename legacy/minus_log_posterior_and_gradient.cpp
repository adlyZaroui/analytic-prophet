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

std::tuple<double, double, Eigen::VectorXd, Eigen::VectorXd> extract_params(const Eigen::Ref<const Eigen::VectorXd>& params) {
    double k = params(0);
    double m = params(1);
    Eigen::VectorXd delta = params.segment(2, 25); // Extract next 25 elements for delta
    Eigen::VectorXd beta = params.tail(params.size() - 27);
    return std::make_tuple(k, m, delta, beta);
}

extern "C" {
    void minus_log_posterior_and_gradient(const double* params,
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
                                          double tau,
                                          double* mlp_out,
                                          double* grad_out);
}

void minus_log_posterior_and_gradient(const double* params,
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
                                      double tau,
                                      double* mlp_out,
                                      double* grad_out) {
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
    Eigen::VectorXd r = normalized_y_vec - y_pred;

    double sum_squared_diff = r.array().square().sum();

    double minus_log_posterior_value = sum_squared_diff / (2 * std::pow(sigma_obs, 2)) + 
                                       std::pow(k, 2) / (2 * std::pow(sigma_k, 2)) + 
                                       std::pow(m, 2) / (2 * std::pow(sigma_m, 2)) + 
                                       beta.array().square().sum() / (2 * std::pow(sigma, 2)) + 
                                       delta.array().abs().sum() / tau;

    // Set minus log posterior value
    *mlp_out = minus_log_posterior_value;

    // Declare grad vector
    Eigen::VectorXd grad(params_size);

    // Compute dk and dm
    grad(0) = -r.dot(t_scaled_vec) / (sigma_obs * sigma_obs) + k / (sigma_k * sigma_k);
    grad(1) = -r.sum() / (sigma_obs * sigma_obs) + m / (sigma_m * sigma_m);

    // Compute ddelta
    Eigen::MatrixXd t_diff = t_scaled_vec.replicate(1, change_points_vec.size()).array().rowwise() - change_points_vec.transpose().array();
    Eigen::MatrixXd delta_contrib = t_diff.array() * A.array();
    Eigen::VectorXd ddelta = -(r.transpose() * delta_contrib).transpose() / (sigma_obs * sigma_obs) + (delta.array().sign() / tau).matrix();
    
    grad.segment(2, delta.size()) = ddelta;

    // Compute dbeta
    int beta_start_index = 2 + delta.size(); // Dynamically calculate the starting index for beta
    Eigen::VectorXd dbeta = -(x.transpose() * r) / (sigma_obs * sigma_obs) + beta / (sigma * sigma);

    grad.segment(beta_start_index, beta.size()) = dbeta;

    std::memcpy(grad_out, grad.data(), grad.size() * sizeof(double));
}

// To compile this file on Mac OS, run:
// g++ -std=c++17 -shared -fPIC -Ofast -o libminus_log_posterior_and_gradient.so minus_log_posterior_and_gradient.cpp -I/opt/homebrew/opt/eigen/include/eigen3