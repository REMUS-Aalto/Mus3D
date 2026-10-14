# Copyright (c) 2026 Veikka Valtteri Kallinen
# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
# Copyright (c) 2024 Kohei Saijo
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: MIT

import itertools

import torch
from torch import nn
from einops import rearrange

class ThresSNRLoss(nn.Module):
    def __init__(self, snr_max=30, solve_perm=True, only_denominator=False, threshold_with="reference"):
        super().__init__()

        assert threshold_with in ["mixture", "reference"]

        self.snr_max = snr_max
        self.only_denominator = only_denominator
        self.temp = 0 if snr_max is None else 10 ** (-snr_max / 10)
        self.threshold_with = threshold_with
        self.solve_perm = solve_perm

    def singlesrc_thresSNR(self, est, ref, eps=1e-8):
        noise = ((ref - est) ** 2).sum(dim=-1)
        tgt = (ref**2).sum(dim=-1)
        threshold = ((ref.sum(dim=-2)) ** 2).sum(dim=-1) if self.threshold_with == "mixture" else tgt

        neg_snr = 10 * torch.log10(noise + self.temp * threshold + eps)
        if not self.only_denominator:
            neg_snr -= 10 * torch.log10(tgt + eps)

        # mean source dimension
        if neg_snr.ndim == 2:
            neg_snr = neg_snr.mean(dim=-1)
        return neg_snr

    def forward(self, est, ref):
        assert est.ndim == ref.ndim, (est.shape, ref.shape)
        return self.singlesrc_thresSNR(est, ref)


class ThresSNRLossWithInactiveSource(nn.Module):
    def __init__(
        self,
        n_src,
        n_ref=None,
        solve_perm=False,
        snr_max=30,
        inactive_thres=-60,
        only_denominator=True,
        zeroref_weight=1.0,
    ):
        super().__init__()
        self.n_src = n_src

        self.solve_perm = solve_perm
        if self.solve_perm:
            if n_ref is None:
                self.perms = list(itertools.permutations(range(n_src)))
            else:
                self.perms = list(itertools.permutations(range(n_src), n_ref))
        else:
            self.perms = [list(range(self.n_src))]

        self.temp = 10 ** (-snr_max / 10) if snr_max is not None else 0
        self.inactive_thres = inactive_thres
        self.only_denominator = only_denominator
        self.zeroref_weight = zeroref_weight

    def l2(self, est, ref):
        return ((est - ref) ** 2).sum(dim=-1)

    def forward(self, est, ref, return_mean=True, return_est=False, return_perm=False, eps=1e-8):
        # NOTE hack since we don't have source dimension in MoisesDB training
        if len(est.shape) < 4:
            est = est.unsqueeze(1)
        if len(ref.shape) < 4:
            ref = ref.unsqueeze(1)

        # for compability with our code
        est = rearrange(est, "b s c t -> b c s t")
        ref = rearrange(ref, "b s c t -> b c s t")

        # est, ref: (n_batch, n_chan, n_src, n_samples)

        mix = ref.sum(dim=-2)

        ref_power = (ref**2).sum(dim=-1)
        mix_power = (mix**2).sum(dim=-1, keepdim=True)

        input_snr = 10 * torch.log10(ref_power / (mix_power + eps))
        activity = input_snr.ge(self.inactive_thres)
        denom_soft_thres = self.temp * (activity * ref_power + (~activity) * mix_power)

        snrs = []
        perms = self.perms
        for p in perms:
            est_permed = est[..., p, :]
            snr = 10 * torch.log10(self.l2(est_permed, ref) + denom_soft_thres + eps)

            if self.zeroref_weight != 1.0:
                snr = snr * (activity + ~activity * self.zeroref_weight)

            if not self.only_denominator:
                num = 10 * torch.log10(activity * ref_power + (~activity) + eps)
                snr = snr - num * activity

            snr = snr.mean(dim=-1)
            snrs.append(snr)
        snrs = torch.stack(snrs, dim=-1)
        loss, idx = torch.min(snrs, dim=-1)

        if return_mean:
            loss = loss.mean()

        if not return_perm:
            if return_est:
                for b in range(est.shape[0]):
                    est[b] = est[b, ..., self.perms[idx[b]], :]
                return loss, est
            else:
                return loss
        else:
            opt_perms = torch.stack(
                [torch.LongTensor(self.perms[idx[b]]) for b in range(est.shape[0])],
                dim=0,
            )  # (n_batch, n_src)
            if return_est:
                for b in range(est.shape[0]):
                    est[b] = est[b, ..., self.perms[idx[b]], :]
                return loss, est, opt_perms
            else:
                return loss, opt_perms


def snr(
    est,
    ref,
    return_mean=True,
    return_est=False,
    return_perm=False,
    negative=True,
    eps=1e-8,
):
    # est, ref: (n_batch, n_chan, n_src, n_samples) or (n_chan, n_src, n_samples)

    n_est = est.shape[-2]
    n_ref = ref.shape[-2]
    assert n_est >= n_ref, (est.shape, ref.shape)

    perms = [list(range(n_est))]

    ref_power = (ref**2).sum(dim=-1)
    snrs = []
    for p in perms:
        est_permed = est[..., p, :]
        snr = 10 * torch.log10(((est_permed - ref) ** 2).sum(dim=-1) + eps)
        snr = snr - 10 * torch.log10(ref_power + eps)

        # snr = snr.mean(dim=-1)
        snrs.append(snr)
    snrs = torch.stack(snrs, dim=-1)
    loss, idx = torch.min(snrs.mean(dim=-2), dim=-1)

    loss = loss.mean() if return_mean else torch.stack([snrs[b, ..., idx[b]] for b in range(idx.shape[0])], dim=0)

    if not negative:
        loss = loss * -1

    if not return_perm:
        if return_est:
            for b in range(est.shape[0]):
                est[b] = est[b, ..., perms[idx[b]], :]
            return loss, est
        else:
            return loss
    else:
        opt_perms = torch.stack(
            [torch.LongTensor(perms[idx[b]]) for b in range(est.shape[0])],
            dim=0,
        )  # (n_batch, n_src)
        if return_est:
            for b in range(est.shape[0]):
                est[b] = est[b, ..., perms[idx[b]], :]
            return loss, est, opt_perms
        else:
            return loss, opt_perms