// Flow-matching UniPC multistep scheduler — C++ port of Ovi's
// FlowUniPCMultistepScheduler (Ovi/ovi/utils/fm_solvers_unipc.py), fixed to the
// config the fusion engine instantiates: num_train_timesteps=1000, solver_order=2,
// solver_type="bh2", prediction_type="flow_prediction", predict_x0=true,
// lower_order_final=true, final_sigmas_type="zero", no thresholding / solver_p /
// disable_corrector / dynamic shifting. State is a flat 1-D float vector
// (float* + n); solver coefficient math in double.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sd {

class UniPCScheduler {
 public:
  explicit UniPCScheduler(int num_train_timesteps = 1000);

  // shift here is the set_timesteps shift (engine passes 5.0); construction
  // shift is fixed to 1 (identity) as in the engine.
  void set_timesteps(int num_inference_steps, double shift);

  // Mirrors python step(model_output, timestep, sample): model_output is the
  // raw velocity prediction at `timestep` for `sample` (n floats); the next
  // sample is written to `out`. `timestep` must be timesteps()[i].
  void step(const float* model_output, int64_t timestep, const float* sample,
            float* out, size_t n);

  int num_inference_steps() const { return num_inference_steps_; }
  const std::vector<float>& sigmas() const { return sigmas_; }
  const std::vector<int64_t>& timesteps() const { return timesteps_; }

 private:
  void init_step_index_(int64_t timestep);
  std::vector<double> convert_model_output_(const float* model_output,
                                            const float* sample, size_t n) const;
  void multistep_uni_p_bh_update_(const std::vector<double>& x, int order,
                                  float* out, size_t n) const;
  std::vector<double> multistep_uni_c_bh_update_(
      const std::vector<double>& this_model_output,
      const std::vector<double>& last_sample,
      const std::vector<double>& this_sample, int order) const;

  int num_train_timesteps_;
  static constexpr int kOrder = 2;  // solver_order
  double sigma_min_ = 0, sigma_max_ = 0;
  std::vector<float> sigmas_;          // len steps+1, last is 0 (float32 parity)
  std::vector<int64_t> timesteps_;     // len steps
  int num_inference_steps_ = -1;
  std::vector<std::vector<double>> model_outputs_;  // ring of kOrder caches
  std::vector<int64_t> timestep_list_;
  int lower_order_nums_ = 0;
  bool has_last_sample_ = false;
  std::vector<double> last_sample_;
  int this_order_ = 0;
  int step_index_ = -1;  // -1 == None
};

}  // namespace sd
