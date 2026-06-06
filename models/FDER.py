import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.topk = configs.topk
        self.revise_len = configs.revise_len
        self.patch_len = self.revise_len
        cfg_stride = int(getattr(configs, 'stride', -1))
        self.stride = cfg_stride if cfg_stride > 0 else self.revise_len // 2

        self.M = int(getattr(configs, 'top_m', 8))
        self.n_low = int(getattr(configs, 'n_low', 3))
        self.n_dom = int(getattr(configs, 'n_dom', 5))

        self.weights_low = nn.Parameter(torch.zeros([1, configs.enc_in, 1]))
        self.weights_dom = nn.Parameter(torch.zeros([1, configs.enc_in, 1]))
        self.fourier_dom_gate = nn.Parameter(torch.tensor(-1.5))
        self.sim_future_gate = nn.Parameter(torch.tensor(-1.5))

        self.tau_evt = float(getattr(configs, 'tau_evt', 3.0))
        self.p_evt = int(getattr(configs, 'p_evt', 4))

        self.dropout = nn.Dropout(float(getattr(configs, 'dropout', 0.0)))

        r = self.revise_len
        H = self.pred_len
        if self.individual:
            self.Linear_Low = nn.ModuleList()
            self.Linear_Dom = nn.ModuleList()
            self.Linear_Sim = nn.ModuleList()
            for _ in range(self.channels):
                ll = nn.Linear(r, H)
                ll.weight = nn.Parameter((1.0 / r) * torch.ones([H, r]))
                self.Linear_Low.append(ll)
                ld = nn.Linear(r, H)
                ld.weight = nn.Parameter((1.0 / r) * torch.ones([H, r]))
                self.Linear_Dom.append(ld)
                ls = nn.Linear((self.topk + 1) * r, H)
                ls.weight = nn.Parameter((1.0 / ((self.topk + 1) * r)) * torch.ones([H, (self.topk + 1) * r]))
                self.Linear_Sim.append(ls)
        else:
            self.Linear_Low = nn.Linear(r, H)
            self.Linear_Low.weight = nn.Parameter((1.0 / r) * torch.ones([H, r]))
            self.Linear_Dom = nn.Linear(r, H)
            self.Linear_Dom.weight = nn.Parameter((1.0 / r) * torch.ones([H, r]))
            self.Linear_Sim = nn.Linear((self.topk + 1) * r, H)
            self.Linear_Sim.weight = nn.Parameter(
                (1.0 / ((self.topk + 1) * r)) * torch.ones([H, (self.topk + 1) * r]))

        self.evt_d = int(getattr(configs, 'evt_d', 8))
        self.evt_in = nn.Linear(r, self.evt_d, bias=False)
        self.evt_out = nn.Linear(self.evt_d, H, bias=True)
        nn.init.zeros_(self.evt_out.weight)
        nn.init.zeros_(self.evt_out.bias)
        self.log_lambda_evt = nn.Parameter(torch.tensor(0.0))
        self.gamma_evt = nn.Parameter(torch.tensor(0.05))

    def patching(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

    def _apply_linear(self, mod, x, B, N, out_dim):
        if self.individual:
            out = torch.zeros(B, N, out_dim, dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                out[:, i, :] = mod[i](x[:, i, :])
            return out
        return mod(x)

    def seq_freq_decompose(self, x_seq):
        n_low, n_dom = self.n_low, self.n_dom
        L = x_seq.shape[-1]
        F_seq = torch.fft.rfft(x_seq, dim=-1)
        d_f = F_seq.shape[-1]

        F_low_g = torch.zeros_like(F_seq)
        F_low_g[..., :n_low] = F_seq[..., :n_low]

        F_high = F_seq[..., n_low:]
        k_dom = min(n_dom, d_f - n_low)
        _, dom_idx_local = torch.topk(torch.abs(F_high), k=k_dom, dim=-1)
        dom_idx = dom_idx_local + n_low
        F_dom_g = torch.zeros_like(F_seq)
        F_dom_g.scatter_(-1, dom_idx, torch.gather(F_seq, -1, dom_idx))

        F_res_g = F_seq - F_low_g - F_dom_g

        seq_low = torch.fft.irfft(F_low_g, n=L, dim=-1)
        seq_dom = torch.fft.irfft(F_dom_g, n=L, dim=-1)
        seq_res = torch.fft.irfft(F_res_g, n=L, dim=-1)
        return seq_low, seq_dom, seq_res

    def freq_similarity_query_aligned(self, F_freq, mode):
        BN, m, d_f = F_freq.shape

        if mode == 'low':
            k = min(self.n_low, d_f)
            idx = torch.arange(k, device=F_freq.device).view(1, -1).expand(BN, -1)
        else:
            if mode == 'dom':
                offset = self.n_low
            elif mode == 'raw':
                offset = 0
            elif mode == 'res':
                offset = min(self.n_low + self.n_dom, d_f - 1)
            else:
                raise ValueError(f"unknown freq sim mode {mode!r}")

            if offset >= d_f:
                offset = 0

            mag_q = torch.abs(F_freq[:, -1, offset:])
            k = min(self.M, mag_q.shape[-1])
            _, top_local = torch.topk(mag_q, k=k, dim=-1)
            idx = top_local + offset

        idx_expand = idx.unsqueeze(1).expand(-1, m, -1)
        bins = torch.gather(F_freq, dim=-1, index=idx_expand)

        if mode == 'res':
            eps = 1e-8
            mag = torch.abs(bins)
            z = bins * (torch.log1p(mag) / (mag + eps))
            z_q = z[:, -1:, :]
            num = (z * z_q.conj()).sum(dim=-1).real
            den = (z.abs().pow(2).sum(dim=-1).sqrt()
                   * z_q.abs().pow(2).sum(dim=-1).sqrt())
            Sim = num / (den + eps)
        else:
            A = torch.log1p(torch.abs(bins))
            A_q = A[:, -1:, :]
            Sim = F.cosine_similarity(A, A_q, dim=-1)

        Sim[:, -1] = float('-inf')
        return Sim

    def _y_low(self, P_low, Sim, B, N):
        r = self.patch_len
        _, topk_idx = torch.topk(Sim, k=self.topk, dim=-1)
        P_ret = torch.gather(P_low, 1, topk_idx.unsqueeze(-1).expand(-1, -1, r))
        P_ret = P_ret.mean(dim=1).reshape(B, N, r)
        P_last = P_low[:, -1, :].reshape(B, N, r)
        W = torch.sigmoid(self.weights_low)
        P_tilde = P_ret * W + P_last * (1.0 - W)
        P_tilde = self.dropout(P_tilde)
        return self._apply_linear(self.Linear_Low, P_tilde, B, N, self.pred_len)

    def _y_dom(self, P_dom, Sim, B, N):
        r = self.patch_len
        H = self.pred_len

        P_last = P_dom[:, -1, :]
        F_last = torch.fft.rfft(P_last, dim=-1)
        d_f = F_last.shape[-1]

        mag = torch.abs(F_last)
        mag[..., 0] = float('-inf')
        n_excl = 1
        if r % 2 == 0:
            mag[..., -1] = float('-inf')
            n_excl = 2
        k_M = min(self.n_dom, d_f - n_excl)
        k_M = max(k_M, 1)
        _, top_idx = torch.topk(mag, k=k_M, dim=-1)
        C_last = torch.gather(F_last, -1, top_idx)

        m = P_dom.shape[1]
        K = max(min(self.topk, m - 1), 1)
        sim_top, topk_idx = torch.topk(Sim, k=K, dim=-1)
        P_ret = torch.gather(P_dom, 1, topk_idx.unsqueeze(-1).expand(-1, -1, r))
        F_ret = torch.fft.rfft(P_ret, dim=-1)
        top_idx_K = top_idx.unsqueeze(1).expand(-1, K, -1)
        M_ret_K = torch.abs(torch.gather(F_ret, -1, top_idx_K))

        sim_safe = torch.where(torch.isfinite(sim_top), sim_top,
                               torch.full_like(sim_top, -1e9))
        w = torch.softmax(sim_safe, dim=-1)
        M_ret = (w.unsqueeze(-1) * M_ret_K).sum(dim=1)

        eps = 1e-8
        M_last = torch.abs(C_last)
        phase_last = C_last / (M_last + eps)
        M_ret_BN = M_ret.reshape(B, N, k_M)
        M_last_BN = M_last.reshape(B, N, k_M)
        phase_BN = phase_last.reshape(B, N, k_M)
        g = torch.sigmoid(self.weights_dom)
        M_cal = g * M_ret_BN + (1.0 - g) * M_last_BN
        C_cal = phase_BN * M_cal

        k_idx = top_idx.reshape(B, N, k_M).to(C_cal.real.dtype)
        t = torch.arange(r, r + H, device=P_dom.device, dtype=k_idx.dtype)
        omega_t = (2.0 * math.pi / r) * k_idx.unsqueeze(-1) * t
        basis = torch.exp(1j * omega_t)
        Y_base = (2.0 / r) * (C_cal.unsqueeze(-1) * basis).sum(dim=-2).real

        P_last_BN = P_last.reshape(B, N, r)
        Y_linear = self._apply_linear(self.Linear_Dom, self.dropout(P_last_BN), B, N, H)

        gate = torch.sigmoid(self.fourier_dom_gate)
        return Y_linear + gate * Y_base

    def _y_sim(self, P_raw, Sim, B, N):
        r = self.patch_len
        H = self.pred_len
        BN, m, _ = P_raw.shape

        _, topk_idx = torch.topk(Sim, k=self.topk, dim=-1)
        P_ret = torch.gather(P_raw, 1, topk_idx.unsqueeze(-1).expand(-1, -1, r))
        P_last = P_raw[:, -1:, :]
        P_cat = torch.cat([P_ret, P_last], dim=1).reshape(B, N, (self.topk + 1) * r)
        P_cat = self.dropout(P_cat)
        Y_linear = self._apply_linear(self.Linear_Sim, P_cat, B, N, H)

        pred_patch = math.ceil(H / r)
        valid_count = m - pred_patch
        if valid_count <= 0:
            return Y_linear

        valid = torch.arange(m, device=P_raw.device).view(1, m) < valid_count
        Sim_future = Sim.masked_fill(~valid, float('-inf'))
        K = max(min(self.topk, valid_count), 1)
        sim_top, anchor_idx = torch.topk(Sim_future, k=K, dim=-1)

        follow_offsets = torch.arange(1, pred_patch + 1, device=P_raw.device)
        follow_idx = anchor_idx.unsqueeze(-1) + follow_offsets.view(1, 1, pred_patch)
        follow_idx_flat = follow_idx.reshape(BN, K * pred_patch)

        P_follow = torch.gather(
            P_raw,
            1,
            follow_idx_flat.unsqueeze(-1).expand(-1, -1, r),
        ).reshape(BN, K, pred_patch * r)[..., :H]

        sim_top_safe = torch.where(
            torch.isfinite(sim_top),
            sim_top,
            torch.full_like(sim_top, -1e9),
        )
        weights = torch.softmax(sim_top_safe, dim=-1)
        Y_future = (weights.unsqueeze(-1) * P_follow).sum(dim=1).reshape(B, N, H)

        gate = torch.sigmoid(self.sim_future_gate)
        return Y_linear + gate * Y_future

    def _y_evt(self, P_res, Sim_in, B, N):
        r = self.patch_len
        H = self.pred_len
        L = self.seq_len
        BN, m, _ = P_res.shape
        eps = 1e-8
        tau_evt = self.tau_evt
        p_evt = self.p_evt
        T_val = min(H, r)
        K_evt = max(min(self.topk, m - 1), 1)

        med_all = P_res.median(dim=-1, keepdim=True).values
        MAD_all = (P_res - med_all).abs().median(dim=-1, keepdim=True).values
        score_all = P_res.abs() / (1.4826 * MAD_all + eps)
        p_actual = min(p_evt, r)
        s_all = score_all.topk(k=p_actual, dim=-1).values.mean(dim=-1)
        tau_all = score_all.argmax(dim=-1)
        s_q = s_all[:, -1]
        tau_q = tau_all[:, -1]
        g_evt = torch.sigmoid(s_q - tau_evt)

        t_q = (m - 1) * self.stride + tau_q
        delta_q = L - t_q

        Sim_evt = Sim_in.masked_fill(s_all <= tau_evt, float('-inf'))
        t_evt_abs = (torch.arange(m, device=P_res.device).view(1, m) * self.stride
                     + tau_all)
        Sim_evt = Sim_evt.masked_fill(
            t_evt_abs + delta_q.unsqueeze(-1) + T_val > L,
            float('-inf')
        )

        R_seq = self._fold_residual(P_res)

        sim_top, i_top = Sim_evt.topk(K_evt, dim=-1)
        has_match = torch.isfinite(sim_top.max(dim=-1).values)

        tau_top = tau_all.gather(1, i_top)
        t_top = i_top * self.stride + tau_top
        value_start = (t_top + delta_q.unsqueeze(-1)).clamp(min=0, max=L - T_val)

        read_idx = value_start.unsqueeze(-1) + torch.arange(T_val, device=R_seq.device).view(1, 1, T_val)
        R_seq_K = R_seq.unsqueeze(1).expand(-1, K_evt, -1)
        V_top = torch.gather(R_seq_K, 2, read_idx)

        sim_top_safe = torch.where(torch.isfinite(sim_top), sim_top,
                                   torch.full_like(sim_top, -1e9))
        weights = torch.softmax(sim_top_safe, dim=-1)
        V_raw = (weights.unsqueeze(-1) * V_top).sum(dim=1)

        med_R = R_seq.median(dim=-1, keepdim=True).values
        MAD_R = (R_seq - med_R).abs().median(dim=-1, keepdim=True).values
        theta = 1.4826 * MAD_R + eps
        lambda_evt = F.softplus(self.log_lambda_evt) + 1e-6
        V_struct = torch.sign(V_raw) * F.relu(V_raw.abs() - lambda_evt * theta)

        if T_val < r:
            V_struct = F.pad(V_struct, (0, r - T_val), mode='constant', value=0.0)

        Y_evt = self.evt_out(self.evt_in(V_struct))

        gate = (self.gamma_evt * g_evt * has_match.float()).unsqueeze(-1)
        Y_evt = gate * Y_evt
        return Y_evt.reshape(B, N, H)

    def _fold_residual(self, P_res):
        BN, _, r = P_res.shape
        L = self.seq_len
        patches_t = P_res.permute(0, 2, 1).contiguous()
        R_seq = F.fold(patches_t, output_size=(1, L), kernel_size=(1, r),
                       stride=(1, self.stride)).reshape(BN, L)
        count = F.fold(torch.ones_like(patches_t), output_size=(1, L),
                       kernel_size=(1, r), stride=(1, self.stride)).reshape(BN, L)
        return R_seq / count.clamp(min=1.0)

    def encoder(self, x_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        B, L, N = x_enc.shape
        x_seq = x_enc.permute(0, 2, 1)

        seq_low, seq_dom, seq_res = self.seq_freq_decompose(x_seq)
        P_low_g = self.patching(seq_low)
        P_dom_g = self.patching(seq_dom)
        P_res_g = self.patching(seq_res)

        Y_pred = torch.zeros(B, N, self.pred_len, dtype=x_enc.dtype, device=x_enc.device)

        F_low_p = torch.fft.rfft(P_low_g, dim=-1)
        Sim_low = self.freq_similarity_query_aligned(F_low_p, 'low')
        Y_pred = Y_pred + self._y_low(P_low_g, Sim_low, B, N)

        F_dom_p = torch.fft.rfft(P_dom_g, dim=-1)
        Sim_dom = self.freq_similarity_query_aligned(F_dom_p, 'dom')
        Y_pred = Y_pred + self._y_dom(P_dom_g, Sim_dom, B, N)

        P_raw = self.patching(x_seq)
        F_raw_p = torch.fft.rfft(P_raw, dim=-1)
        Sim_raw = self.freq_similarity_query_aligned(F_raw_p, 'raw')
        Y_pred = Y_pred + self._y_sim(P_raw, Sim_raw, B, N)

        F_res_p = torch.fft.rfft(P_res_g, dim=-1)
        Sim_res = self.freq_similarity_query_aligned(F_res_p, 'res')
        Y_pred = Y_pred + self._y_evt(P_res_g, Sim_res, B, N)

        return Y_pred.permute(0, 2, 1) * std_enc + mean_enc

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]
