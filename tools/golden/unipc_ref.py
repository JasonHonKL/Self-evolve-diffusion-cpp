#!/usr/bin/env python3
"""Golden reference for src/sampler/unipc.cpp: numpy transcription of Ovi's
FlowUniPCMultistepScheduler (Ovi/ovi/utils/fm_solvers_unipc.py) for the exact
config the fusion engine uses (ovi_fusion_engine.get_scheduler_time_steps):
FlowUniPCMultistepScheduler(1000, shift=1, use_dynamic_shifting=False) +
set_timesteps(steps, shift=5.0). No torch/diffusers. Fake velocity model
v(x,t) = 0.1*sin(x + t); dumps x0, sigma schedule and every step's sample to
unipc_traj.npz (np.savez STORED, f32 entries — readable by src/test_support/npz.cpp).

Caches float32 like torch does; coefficient math in float64.
"""
import os

import numpy as np

N = 8
STEPS = 10
SHIFT = 5.0
NUM_TRAIN = 1000


class RefUniPC:
    """FlowUniPCMultistepScheduler with solver_order=2, bh2, predict_x0,
    flow_prediction, lower_order_final, final_sigmas_type='zero'."""

    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = 2
        self.predict_x0 = True
        self.solver_type = "bh2"
        self.lower_order_final = True
        alphas = np.linspace(1, 1 / num_train_timesteps,
                             num_train_timesteps)[::-1].copy()
        sigmas = (1.0 - alphas).astype(np.float32)
        # construction shift = 1 -> identity
        self.sigma_min = float(sigmas[-1])  # 0.0
        self.sigma_max = float(sigmas[0])   # float32(0.999)

    def set_timesteps(self, num_inference_steps, shift):
        sigmas = np.linspace(self.sigma_max, self.sigma_min,
                             num_inference_steps + 1)[:-1].copy()
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigma_last = 0.0  # final_sigmas_type == "zero"
        timesteps = sigmas * self.num_train_timesteps
        self.sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        self.timesteps = timesteps.astype(np.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.step_index = None
        self.this_order = None

    def index_for_timestep(self, timestep):
        indices = np.nonzero(self.timesteps == timestep)[0]
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos])

    def convert_model_output(self, model_output, sample):
        # flow_prediction + predict_x0
        sigma_t = float(self.sigmas[self.step_index])
        x0_pred = sample.astype(np.float64) - sigma_t * model_output.astype(np.float64)
        return x0_pred.astype(np.float32)

    def _rks_D1s_R_b(self, order, si_offset):
        m0 = self.model_outputs[-1].astype(np.float64)
        sigma_t = float(self.sigmas[self.step_index + 1 - si_offset])
        sigma_s0 = float(self.sigmas[self.step_index - si_offset])
        alpha_t, alpha_s0 = 1 - sigma_t, 1 - sigma_s0
        lambda_t = np.log(alpha_t) - np.log(sigma_t)
        lambda_s0 = np.log(alpha_s0) - np.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks, D1s = [], []
        for i in range(1, order):
            si = self.step_index - i - si_offset
            mi = self.model_outputs[-(i + 1)].astype(np.float64)
            alpha_si, sigma_si = 1 - float(self.sigmas[si]), float(self.sigmas[si])
            lambda_si = np.log(alpha_si) - np.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)

        R, b = [], []
        hh = -h if self.predict_x0 else h
        h_phi_1 = np.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = hh if self.solver_type == "bh1" else np.expm1(hh)
        for i in range(1, order + 1):
            R.append([rk ** (i - 1) for rk in rks])
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        return sigma_t, sigma_s0, alpha_t, h_phi_1, B_h, R, b, D1s

    def multistep_uni_p_bh_update(self, sample, order):
        m0 = self.model_outputs[-1].astype(np.float64)
        x = sample.astype(np.float64)
        (sigma_t, sigma_s0, alpha_t, h_phi_1, B_h, R, b,
         D1s) = self._rks_D1s_R_b(order, si_offset=0)

        if len(D1s) > 0:
            if order == 2:
                rhos_p = np.array([0.5])
            else:
                m = order - 1
                rhos_p = np.linalg.solve(np.array(R)[:m, :m], np.array(b)[:m])
        else:
            rhos_p = None

        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if rhos_p is not None:
            pred_res = sum(rhos_p[k] * D1s[k] for k in range(len(D1s)))
            x_t = x_t - alpha_t * B_h * pred_res
        return x_t.astype(np.float32)

    def multistep_uni_c_bh_update(self, this_model_output, last_sample,
                                  this_sample, order):
        m0 = self.model_outputs[-1].astype(np.float64)
        x = last_sample.astype(np.float64)
        model_t = this_model_output.astype(np.float64)
        (sigma_t, sigma_s0, alpha_t, h_phi_1, B_h, R, b,
         D1s) = self._rks_D1s_R_b(order, si_offset=1)

        if order == 1:
            rhos_c = np.array([0.5])
        else:
            rhos_c = np.linalg.solve(np.array(R), np.array(b))

        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        corr_res = sum(rhos_c[k] * D1s[k] for k in range(len(D1s)))
        D1_t = model_t - m0
        x_t = x_t - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t.astype(np.float32)

    def step(self, model_output, timestep, sample):
        if self.step_index is None:
            self.step_index = self.index_for_timestep(timestep)

        use_corrector = self.step_index > 0 and self.last_sample is not None

        model_output_convert = self.convert_model_output(model_output, sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order)

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order,
                             len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(sample, self.this_order)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self.step_index += 1
        return prev_sample


def main():
    rng = np.random.default_rng(1234)
    x0 = (rng.standard_normal(N) * 0.5).astype(np.float32)

    sched = RefUniPC(NUM_TRAIN)
    sched.set_timesteps(STEPS, SHIFT)

    x = x0.copy()
    traj = []
    with np.errstate(divide="ignore", invalid="ignore"):  # final sigma=0 -> log(0)
        for t in sched.timesteps:
            mo = (0.1 * np.sin(x.astype(np.float64) + float(t))).astype(np.float32)
            x = sched.step(mo, int(t), x)
            traj.append(x.astype(np.float64))

    arrays = {"x0": x0, "sigmas": sched.sigmas}
    arrays.update({f"s{i}": traj[i].astype(np.float32)
                   for i in range(len(traj))})

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "unipc_traj.npz")
    np.savez(out, **arrays)  # STORED zip entries only — npz.cpp limit

    chk = np.load(out)  # self-check: readable + consistent
    assert chk["x0"].shape == (N,) and chk["s0"].shape == (N,)
    assert len(traj) == STEPS and chk["sigmas"].shape == (STEPS + 1,)
    assert np.array_equal(chk[f"s{STEPS-1}"], traj[-1].astype(np.float32))
    print(f"wrote {out}: x0 + {STEPS} steps (n={N}), shift={SHIFT}")
    print("timesteps:", sched.timesteps.tolist())
    print("final sample:", np.array2string(traj[-1], precision=6))


if __name__ == "__main__":
    main()
