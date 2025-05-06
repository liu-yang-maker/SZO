import copy
import math
import torch
from torch.optim.optimizer import Optimizer, required

class SZO(Optimizer):
    r"""
    Zeroth‑Order Sharpness‑Aware Minimization (ZO‑SAM)

    实现在当前参数处用随机向量做有限差分，近似梯度，
    然后沿该近似梯度做 ρ‑范数扰动，再在扰动点上重新做 ZO 估计，
    最终按 SAM 思路做更新。

    Args:
        params (iterable): 待优化参数
        lr (float): 学习率 η
        rho (float): SAM 的 neighborhood size ρ
        epsilon (float): 有限差分步长 ε
        q (int): 随机方向数量（RGE 版），越大方差越小
        two_sided (bool): 若为 True 则使用对称差分 (f(x+εu)-f(x-εu))/(2ε)
        estimator (str): "rge" | "cge"。cge 会走坐标方向，
                         rge 使用随机高斯方向（推荐）
        weight_decay (float): L2 正则
        seed (int or None): 固定随机种子方便复现
    Usage:
        >>> optim = SZO(model.parameters(), lr=1e-3, rho=0.05)
        >>> def closure():
        ...     logits = model(x)
        ...     return loss_fn(logits, y)          # forward 但不 backprop
        >>> optim.step(closure)
    """
    def __init__(self, params, lr=required, rho=0.05, epsilon=1e-3,
                 q=20, two_sided=True, estimator="rge",
                 weight_decay=0.0, seed=None):

        if lr is not required and lr <= 0.0:
            raise ValueError("Invalid lr: {}".format(lr))
        if rho <= 0.0:
            raise ValueError("Invalid rho: {}".format(rho))
        if epsilon <= 0.0:
            raise ValueError("Invalid epsilon: {}".format(epsilon))
        if q < 1:
            raise ValueError("q must be >= 1")

        defaults = dict(lr=lr, rho=rho, eps=epsilon, q=q, two_sided=two_sided,
                        estimator=estimator.lower(), weight_decay=weight_decay)

        super().__init__(params, defaults)

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    # ------------------------------------------------------------
    # 核心：一次 ZO 梯度估计（支持单向或双向差分，随机/坐标）
    # ------------------------------------------------------------
    @torch.no_grad()
    def _zo_grad(self, closure, params, base_loss,
                 eps, q, two_sided, estimator):
        """返回与 params 形状一致的梯度估计列表"""
        grads = [torch.zeros_like(p) for p in params]   # 累积器
        if estimator == "cge":       # 坐标差分（成本大，不建议太大模型用）
            if q != 1:
                raise ValueError("CGE 不需要 q>1；将使用完整坐标扫描")
            for idx, p in enumerate(params):
                flat_dim = p.numel()
                p_view = p.view(-1)
                for j in range(flat_dim):
                    e = torch.zeros_like(p_view)
                    e[j] = 1.0
                    p_view.add_(eps * e)
                    loss_plus = closure()
                    if two_sided:
                        p_view.sub_(2 * eps * e)
                        loss_minus = closure()
                        grad_scalar = (loss_plus - loss_minus) / (2 * eps)
                    else:
                        grad_scalar = (loss_plus - base_loss) / eps
                    p_view.add_(eps * e)  # 复原
                    grads[idx].view(-1)[j] = grad_scalar
        else:                         # RGE：随机高斯/拉德马赫方向
            for _ in range(q):
                directions = []
                # 1. 对所有参数生成等长随机向量 u
                for p in params:
                    u = torch.randn_like(p, generator=self._rng)
                    u = u / (u.norm().clamp_min(1e-12))
                    directions.append(u)

                    # 添加扰动 θ+εu
                    p.add_(eps * u)

                loss_plus = closure()         # f(θ+εu)

                if two_sided:
                    # θ-εu
                    for p, u in zip(params, directions):
                        p.add_(-2 * eps * u)
                    loss_minus = closure()    # f(θ-εu)
                    coeff = (loss_plus - loss_minus) / (2 * eps)
                    # 复原到 θ
                    for p, u in zip(params, directions):
                        p.add_(eps * u)
                else:
                    coeff = (loss_plus - base_loss) / eps
                    for p, u in zip(params, directions):
                        p.add_(-eps * u)     # 复原

                # ∑ coeff * u
                for g, u in zip(grads, directions):
                    g.add_(coeff * u)

            # 平均
            grads = [g.div_(q) for g in grads]

        return grads

    # ------------------------------------------------------------
    # public: optimizer step
    # ------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure):
        """
        参数更新 = SAM 两阶段 ZO。
        需要用户传入的 closure 只做前向，返回标量 loss，
        不做 .backward()。
        """
        if closure is None:
            raise RuntimeError("ZO‑SAM requires closure that returns loss.")

        # --- 0. 记录 base loss ---
        base_loss = closure()

        for group in self.param_groups:
            eps   = group['eps']
            rho   = group['rho']
            q     = group['q']
            two_sided = group['two_sided']
            est   = group['estimator']
            wd    = group['weight_decay']
            lr    = group['lr']

            params = group['params']

            # 1) ZO 估计 ε_t (在 θ)
            grad_eps = self._zo_grad(closure, params, base_loss,
                                     eps, q, two_sided, est)

            # 2) 生成 SAM 扰动 θ̃ = θ + ρ·ε_t/‖ε_t‖
            # 首先求全局 norm
            sq_sum = sum(torch.sum(g**2) for g in grad_eps)
            grad_norm = math.sqrt(sq_sum.item()) + 1e-12
            scale = rho / grad_norm
            # 保存扰动向量，便于后面做真正 update
            perturbs = []
            for p, g in zip(params, grad_eps):
                delta = scale * g
                p.add_(delta)        # θ ← θ + δ
                perturbs.append(delta.clone())

            # 3) 在 θ̃ 上再做一次 ZO 估计 Δ̂_t
            loss_tilde = closure()   # f(θ̃)
            grad_hat = self._zo_grad(closure, params, loss_tilde,
                                     eps, q, two_sided, est)

            # 4) 把参数拉回中心点 θ 再做梯度下降
            for p, delta, ghat in zip(params, perturbs, grad_hat):
                p.sub_(delta)                      # 回到 θ
                # weight decay
                if wd != 0:
                    ghat.add_(wd, p)
                p.add_(-lr, ghat)                  # θ ← θ - η·Δ̂

        return base_loss
