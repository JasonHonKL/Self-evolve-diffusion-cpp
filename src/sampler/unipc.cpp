// Flow-matching UniPC multistep scheduler, port of Ovi's
// FlowUniPCMultistepScheduler (fm_solvers_unipc.py) — Ovi's exact config.
// Control flow mirrors the python 1:1 (same caches, same call order).
#include "sampler/unipc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace sd {
namespace {

// Gaussian elimination with partial pivoting, in-place; b becomes the solution.
bool solve_linear(std::vector<double>& a, std::vector<double>& b, int n) {
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::fabs(a[(size_t)r * n + col]) > std::fabs(a[(size_t)piv * n + col]))
        piv = r;
    if (a[(size_t)piv * n + col] == 0.0) return false;
    if (piv != col) {
      for (int c = 0; c < n; ++c)
        std::swap(a[(size_t)col * n + c], a[(size_t)piv * n + c]);
      std::swap(b[col], b[piv]);
    }
    for (int r = col + 1; r < n; ++r) {
      double f = a[(size_t)r * n + col] / a[(size_t)col * n + col];
      for (int c = col; c < n; ++c)
        a[(size_t)r * n + c] -= f * a[(size_t)col * n + c];
      b[r] -= f * b[col];
    }
  }
  for (int r = n - 1; r >= 0; --r) {
    for (int c = r + 1; c < n; ++c) b[r] -= a[(size_t)r * n + c] * b[c];
    b[r] /= a[(size_t)r * n + r];
  }
  return true;
}

// lambda = log(alpha) - log(sigma) with alpha = 1 - sigma (flow convention).
double lambda_of(double sigma) {
  return std::log(1.0 - sigma) - std::log(sigma);
}

}  // namespace

UniPCScheduler::UniPCScheduler(int num_train_timesteps)
    : num_train_timesteps_(num_train_timesteps) {
  // init: alphas = linspace(1, 1/n, n)[::-1]; sigmas = 1 - alphas (float32);
  // construction shift = 1 -> identity. Only the endpoints are used later.
  sigma_max_ = (double)(float)(1.0 - 1.0 / num_train_timesteps);  // sigmas[0]
  sigma_min_ = (double)(float)(1.0 - 1.0);                        // sigmas[-1]
}

void UniPCScheduler::set_timesteps(int num_inference_steps, double shift) {
  // sigmas = np.linspace(sigma_max, sigma_min, steps+1)[:-1], numpy op order,
  // then shift * s / (1 + (shift-1) * s), then astype(float32); last sigma 0.
  int num = num_inference_steps + 1;
  double step = (sigma_min_ - sigma_max_) / (num - 1);
  std::vector<double> sig;
  for (int i = 0; i + 1 < num; ++i) sig.push_back(i * step + sigma_max_);
  for (double& s : sig) s = shift * s / (1 + (shift - 1) * s);
  timesteps_.clear();
  sigmas_.clear();
  for (double s : sig) {
    timesteps_.push_back((int64_t)(s * num_train_timesteps_));  // .to(int64): trunc
    sigmas_.push_back((float)s);                                // astype(float32)
  }
  sigmas_.push_back(0.0f);  // final_sigmas_type == "zero"
  num_inference_steps_ = (int)timesteps_.size();
  model_outputs_.assign(kOrder, {});
  timestep_list_.assign(kOrder, 0);
  lower_order_nums_ = 0;
  has_last_sample_ = false;
  last_sample_.clear();
  this_order_ = 0;
  step_index_ = -1;
}

void UniPCScheduler::init_step_index_(int64_t timestep) {
  std::vector<int> idx;
  for (size_t i = 0; i < timesteps_.size(); ++i)
    if (timesteps_[i] == timestep) idx.push_back((int)i);
  if (idx.empty()) throw std::runtime_error("unipc: timestep not in schedule");
  step_index_ = idx[idx.size() > 1 ? 1 : 0];
}

std::vector<double> UniPCScheduler::convert_model_output_(
    const float* model_output, const float* sample, size_t n) const {
  // flow_prediction + predict_x0: x0_pred = sample - sigma_t * model_output
  double sigma_t = (double)sigmas_[step_index_];
  std::vector<double> x0(n);
  for (size_t j = 0; j < n; ++j)
    x0[j] = (double)(float)((double)sample[j] -
                            sigma_t * (double)model_output[j]);  // f32 cache parity
  return x0;
}

void UniPCScheduler::multistep_uni_p_bh_update_(const std::vector<double>& x,
                                                int order, float* out,
                                                size_t n) const {
  const std::vector<double>& m0 = model_outputs_.back();
  double sigma_t = (double)sigmas_[step_index_ + 1];
  double sigma_s0 = (double)sigmas_[step_index_];
  double alpha_t = 1 - sigma_t;
  double lambda_t = lambda_of(sigma_t), lambda_s0 = lambda_of(sigma_s0);
  double h = lambda_t - lambda_s0;

  std::vector<double> rks, b;
  std::vector<std::vector<double>> d1s;
  for (int i = 1; i < order; ++i) {
    int si = step_index_ - i;
    const std::vector<double>& mi = model_outputs_[model_outputs_.size() - 1 - i];
    double rk = (lambda_of((double)sigmas_[si]) - lambda_s0) / h;
    rks.push_back(rk);
    std::vector<double> d(n);
    for (size_t j = 0; j < n; ++j) d[j] = (mi[j] - m0[j]) / rk;
    d1s.push_back(std::move(d));
  }
  rks.push_back(1.0);

  double hh = -h;  // predict_x0
  double h_phi_1 = std::expm1(hh);
  double h_phi_k = h_phi_1 / hh - 1;
  double B_h = std::expm1(hh);  // solver_type == "bh2"
  double factorial_i = 1;
  std::vector<double> R((size_t)order * order);
  for (int i = 1; i <= order; ++i) {
    for (int k = 0; k < order; ++k)
      R[(size_t)(i - 1) * order + k] = std::pow(rks[k], i - 1);
    b.push_back(h_phi_k * factorial_i / B_h);
    factorial_i *= i + 1;
    h_phi_k = h_phi_k / hh - 1 / factorial_i;
  }

  std::vector<double> rhos_p;
  if (!d1s.empty()) {
    if (order == 2) {
      rhos_p = {0.5};
    } else {
      int m = order - 1;  // solve R[:-1,:-1] x = b[:-1]
      std::vector<double> a((size_t)m * m), bb(m);
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < m; ++c) a[(size_t)r * m + c] = R[(size_t)r * order + c];
        bb[r] = b[r];
      }
      if (!solve_linear(a, bb, m)) throw std::runtime_error("unipc: singular R");
      rhos_p = std::move(bb);
    }
  }

  double c1 = sigma_t / sigma_s0, c2 = alpha_t * h_phi_1;
  for (size_t j = 0; j < n; ++j) {
    double v = c1 * x[j] - c2 * m0[j];
    if (!d1s.empty()) {
      double pr = 0;
      for (size_t k = 0; k < d1s.size(); ++k) pr += rhos_p[k] * d1s[k][j];
      v -= alpha_t * B_h * pr;
    }
    out[j] = (float)v;
  }
}

std::vector<double> UniPCScheduler::multistep_uni_c_bh_update_(
    const std::vector<double>& model_t, const std::vector<double>& last_sample,
    const std::vector<double>& this_sample, int order) const {
  size_t n = this_sample.size();
  const std::vector<double>& m0 = model_outputs_.back();
  double sigma_t = (double)sigmas_[step_index_];
  double sigma_s0 = (double)sigmas_[step_index_ - 1];
  double alpha_t = 1 - sigma_t;
  double lambda_t = lambda_of(sigma_t), lambda_s0 = lambda_of(sigma_s0);
  double h = lambda_t - lambda_s0;

  std::vector<double> rks, b;
  std::vector<std::vector<double>> d1s;
  for (int i = 1; i < order; ++i) {
    int si = step_index_ - (i + 1);
    const std::vector<double>& mi = model_outputs_[model_outputs_.size() - 1 - i];
    double rk = (lambda_of((double)sigmas_[si]) - lambda_s0) / h;
    rks.push_back(rk);
    std::vector<double> d(n);
    for (size_t j = 0; j < n; ++j) d[j] = (mi[j] - m0[j]) / rk;
    d1s.push_back(std::move(d));
  }
  rks.push_back(1.0);

  double hh = -h;  // predict_x0
  double h_phi_1 = std::expm1(hh);
  double h_phi_k = h_phi_1 / hh - 1;
  double B_h = std::expm1(hh);  // solver_type == "bh2"
  double factorial_i = 1;
  std::vector<double> R((size_t)order * order);
  for (int i = 1; i <= order; ++i) {
    for (int k = 0; k < order; ++k)
      R[(size_t)(i - 1) * order + k] = std::pow(rks[k], i - 1);
    b.push_back(h_phi_k * factorial_i / B_h);
    factorial_i *= i + 1;
    h_phi_k = h_phi_k / hh - 1 / factorial_i;
  }

  std::vector<double> rhos_c;
  if (order == 1) {
    rhos_c = {0.5};
  } else {  // solve R x = b (full)
    std::vector<double> a(R), bb(b);
    if (!solve_linear(a, bb, order)) throw std::runtime_error("unipc: singular R");
    rhos_c = std::move(bb);
  }

  double c1 = sigma_t / sigma_s0, c2 = alpha_t * h_phi_1, c3 = alpha_t * B_h;
  std::vector<double> xt(n);
  for (size_t j = 0; j < n; ++j) {
    double v = c1 * last_sample[j] - c2 * m0[j];
    double corr = 0;
    for (size_t k = 0; k < d1s.size(); ++k) corr += rhos_c[k] * d1s[k][j];
    v -= c3 * (corr + rhos_c.back() * (model_t[j] - m0[j]));
    xt[j] = v;
  }
  return xt;
}

void UniPCScheduler::step(const float* model_output, int64_t timestep,
                          const float* sample, float* out, size_t n) {
  if (num_inference_steps_ < 0)
    throw std::runtime_error("unipc: call set_timesteps first");
  if (step_index_ < 0) init_step_index_(timestep);

  // disable_corrector is empty in Ovi's config
  bool use_corrector = step_index_ > 0 && has_last_sample_;

  std::vector<double> moc = convert_model_output_(model_output, sample, n);
  std::vector<double> x(sample, sample + n);
  if (use_corrector)
    x = multistep_uni_c_bh_update_(moc, last_sample_, x, this_order_);

  for (int i = 0; i < kOrder - 1; ++i) {
    model_outputs_[i] = std::move(model_outputs_[i + 1]);
    timestep_list_[i] = timestep_list_[i + 1];
  }
  model_outputs_[kOrder - 1] = std::move(moc);
  timestep_list_[kOrder - 1] = timestep;

  int this_order =
      std::min(kOrder, (int)timesteps_.size() - step_index_);  // lower_order_final
  this_order_ = std::min(this_order, lower_order_nums_ + 1);

  last_sample_ = x;
  has_last_sample_ = true;
  multistep_uni_p_bh_update_(x, this_order_, out, n);

  if (lower_order_nums_ < kOrder) ++lower_order_nums_;
  ++step_index_;
}

}  // namespace sd
