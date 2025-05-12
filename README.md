# Adaptive-MCMC
Adaptive Markov Chain Monte Carlo (MCMC) algorithms

## Examples

```{python}
dim = 20
cov = 0.5 * np.eye(dim) + 0.5
inv_cov = np.linalg.inv(cov)

def logp(x):
    return multivariate_normal.logpdf(x, mean=np.zeros(dim), cov=cov)

def grad_logp(x):
    return -inv_cov @ x

mala = AdaptiveMALA(logp, grad_logp, initial_sample=np.zeros(dim), eps0=0.5)
samples, acc, eps_hist, esjd_hist = mala.run(20_000)
print(f"Acceptance: {acc:.3f}, Final epsilon = {eps_hist[-1]:.4f}")
```
