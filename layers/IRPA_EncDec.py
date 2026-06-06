import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
import math
import numpy as np

class IRPA(nn.Module):

    def __init__(self, configs, d_model=None):
        super(IRPA, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decomposition = series_decomp(configs.moving_avg)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.topk = configs.topk
        self.revise_len = configs.revise_len
        self.patch_len = self.revise_len
        self.freq_top_m = max(int(getattr(configs, 'top_m', 0)), 0)
        if d_model != None:
            self.d_model = d_model
        else:
            self.d_model = self.revise_len
        self.weights_trend = torch.nn.Parameter(torch.ones([configs.enc_in, 1]))
        self.weights_seasonal = torch.nn.Parameter(torch.ones([configs.enc_in, 1])) 
        self.stride = self.revise_len // 2

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList() # Seasonal Refinement
            self.Linear_Trend = nn.ModuleList() # Trend Refinement
            self.Linear_HSA = nn.ModuleList() # Historical Similarity Auxiliary
            self.Linear_HSPPA = nn.ModuleList() # Historical Seasonal Pattern Prediction Auxiliary

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.revise_len, self.d_model))
                self.Linear_Trend.append(
                    nn.Linear(self.revise_len, self.d_model))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.revise_len) * torch.ones([self.d_model, self.revise_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.revise_len) * torch.ones([self.d_model, self.revise_len]))

                self.Linear_HSA.append(nn.Linear((self.topk + 1) * self.revise_len, self.pred_len))
                self.Linear_HSA[i].weight = nn.Parameter(
                    (1 / ((self.topk + 1) * self.revise_len)) * torch.ones([self.pred_len, (self.topk + 1) * self.revise_len]))
                self.Linear_HSPPA.append(nn.Linear(self.pred_len, self.pred_len))
                self.Linear_HSPPA[i].weight = nn.Parameter(
                    (1 / self.pred_len) * torch.ones([self.pred_len, self.pred_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.revise_len, self.d_model)
            self.Linear_Trend = nn.Linear(self.revise_len, self.d_model)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.revise_len) * torch.ones([self.d_model, self.revise_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.revise_len) * torch.ones([self.d_model, self.revise_len]))

            self.Linear_HSA = nn.Linear((self.topk + 1) * self.revise_len, self.pred_len)
            self.Linear_HSA.weight = nn.Parameter(
                (1 / ((self.topk + 1) * self.revise_len)) * torch.ones([self.pred_len, (self.topk + 1) * self.revise_len]))
            self.Linear_HSPPA = nn.Linear(self.pred_len, self.pred_len)
            self.Linear_HSPPA.weight = nn.Parameter(
                (1 / self.pred_len) * torch.ones([self.pred_len, self.pred_len]))

    def patching(self, x):
        # x: B, N, L
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return x

    def calculate_similarity(self, s, f, method=None):
        if method == "hellinger_distance":
            s = torch.softmax(s, -1)
            f = torch.softmax(f, -1)
            sf = torch.sum(torch.sqrt(s * f), dim=2)
            return torch.sqrt(1 - sf)
        elif method == "manhattan_distance":
            return torch.norm(s - f, p=1, dim=2)
        elif method == "euclidean_distance":
            return torch.norm(s - f, p=2, dim=2)
        elif method == "cosine_similarity":
            return torch.nn.functional.cosine_similarity(s, f, dim=2) 
        else: # pearson correlation
            mean_s = torch.mean(s, dim=2, keepdim=True)
            mean_f = torch.mean(f, dim=2, keepdim=True)
            cov_sf = torch.mean((s - mean_s) * (f - mean_f), dim=2)
            return cov_sf / (torch.std(s, dim=2) * torch.std(f, dim=2))

    def calculate_frequency_similarity(self, s, f=None):
        """
        Compare patches in the frequency domain for retrieval.

        Uses the log amplitude spectrum of each patch as the retrieval key. The
        DC bin is removed so the score emphasizes periodic shape instead of the
        patch mean. If --top_m is set, only the query patch's strongest bins are
        used, keeping all candidate keys aligned to the same frequencies.
        """
        if f is None:
            f = s[:, -1:, :]

        s_mag = torch.log1p(torch.abs(torch.fft.rfft(s.float(), dim=-1)))
        f_mag = torch.log1p(torch.abs(torch.fft.rfft(f.float(), dim=-1)))

        if s_mag.size(-1) > 1:
            s_mag = s_mag[..., 1:]
            f_mag = f_mag[..., 1:]

        if s_mag.size(-1) == 0:
            return torch.zeros(s.size(0), s.size(1), dtype=s.dtype, device=s.device)

        if 0 < self.freq_top_m < s_mag.size(-1):
            _, top_idx = torch.topk(f_mag, k=self.freq_top_m, dim=-1)
            s_mag = torch.gather(s_mag, -1, top_idx.expand(-1, s_mag.size(1), -1))
            f_mag = torch.gather(f_mag, -1, top_idx)

        return F.cosine_similarity(s_mag, f_mag, dim=-1, eps=1e-8).to(s.dtype)

    def periodic_scale(self, i, m):
        """
        Returns:
        The scaled value in the range [0.9, 1], with the values at the start and end indices being 1.
        """
        if m <= 1:
            return torch.ones_like(i, dtype=torch.float32)
        x = i / (m - 1)
        x = x * int(m/5)
        return 1 - 1/10 * torch.sin(np.pi * x)**2

    def forward(self, x_enc):
        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach() 
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        B, L, N = x_enc.shape
        seasonal_init, trend_init = self.decomposition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_init = self.patching(seasonal_init)
        trend_init = self.patching(trend_init)
        x_patch = self.patching(x_enc.permute(0, 2, 1))

        cal_sim = self.calculate_frequency_similarity(seasonal_init, seasonal_init[:,-1,:].unsqueeze(1))
        # Apply the periodic scaling function
        cal_sim = cal_sim * self.periodic_scale(torch.arange(cal_sim.size(-1), device=cal_sim.device).unsqueeze(0), cal_sim.size(-1)) 
        weights_trend = torch.sigmoid(self.weights_trend)
        weights_seasonal = torch.sigmoid(self.weights_seasonal)

        _, max_index = torch.max(cal_sim[:,:-1], dim=1)
        # Use the gather method to obtain the patch most similar
        seasonal_refinement = seasonal_init.gather(1, max_index.view(-1, 1, 1).expand(-1, -1, self.revise_len)).squeeze(1)
        seasonal_refinement = torch.reshape(seasonal_refinement + seasonal_init[:,0,:], (B, N, self.revise_len))
        seasonal_refinement = torch.reshape(seasonal_init[:,-1,:], (B, N, self.revise_len))*weights_seasonal + seasonal_refinement*(1-weights_seasonal) 

        _, topk_indices = torch.topk(cal_sim[:,:-1], self.topk, dim=1)
        # Use these indices to select the k most similar patches
        trend_refinement = torch.gather(trend_init, 1, topk_indices.unsqueeze(-1).expand(-1, -1, self.revise_len))
        trend_refinement = torch.reshape(torch.mean(torch.sigmoid(trend_refinement), dim=1, keepdim=True), (B, N, self.revise_len))
        trend_refinement = trend_refinement*(1-weights_trend) + torch.reshape(trend_init[:,-1,:], (B, N, self.revise_len))*weights_trend
        
        similarity_auxiliary = torch.gather(x_patch, 1, topk_indices.unsqueeze(-1).expand(-1, -1, self.revise_len))
        similarity_auxiliary = torch.reshape(torch.cat([similarity_auxiliary, x_patch[:,-1,:].unsqueeze(1)], 1), (B, N, (self.topk + 1) * self.revise_len))

        pred_patch = math.ceil(self.pred_len / self.patch_len)
        _, max_index = torch.max(cal_sim[:,:-pred_patch], dim=1) 
        # Create an index tensor containing all indices from maxindex + 1 to maxindex + predpatch
        indices = max_index.unsqueeze(1) + torch.arange(1, pred_patch + 1).to(max_index.device).unsqueeze(0)
        prediction_auxiliary = seasonal_init.gather(1, indices.unsqueeze(-1).expand(-1, -1, self.revise_len)) 
        prediction_auxiliary = torch.reshape(prediction_auxiliary, (B, N, -1))[:,:,:self.pred_len]

        if self.individual:
            seasonal_output = torch.zeros([seasonal_refinement.size(0), seasonal_refinement.size(1), self.d_model],
                                          dtype=seasonal_refinement.dtype).to(seasonal_refinement.device)
            trend_output = torch.zeros([trend_refinement.size(0), trend_refinement.size(1), self.d_model],
                                       dtype=trend_refinement.dtype).to(trend_refinement.device)
            prediction_output = torch.zeros([prediction_auxiliary.size(0), prediction_auxiliary.size(1), self.pred_len],
                                          dtype=prediction_auxiliary.dtype).to(prediction_auxiliary.device)
            similarity_output = torch.zeros([similarity_auxiliary.size(0), similarity_auxiliary.size(1), self.pred_len],
                                       dtype=similarity_auxiliary.dtype).to(similarity_auxiliary.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_refinement[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_refinement[:, i, :])
                prediction_output[:, i, :] = self.Linear_HSPPA[i](
                    prediction_auxiliary[:, i, :])
                similarity_output[:, i, :] = self.Linear_HSA[i](
                    similarity_auxiliary[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_refinement)
            trend_output = self.Linear_Trend(trend_refinement)
            prediction_output = self.Linear_HSPPA(prediction_auxiliary)
            similarity_output = self.Linear_HSA(similarity_auxiliary)

        IR = seasonal_output + trend_output
        PA = prediction_output + similarity_output

        return IR, PA, mean_enc, std_enc
