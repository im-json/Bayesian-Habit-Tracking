# Bayesian Habit Tracking
Autoregressive Poisson model with examples

**Premise**

Each week I count how many instances of bad habits I engage in. These instances are defined by units, such as consuming 60 milligrams of caffeine.

**4-week AR(4) model**

Let $y$ be instances of bad habits for a given week.

$$
y^{\mathrm{T}} =
\begin{bmatrix}
7 & 6 & 0 & 0 & 0 & 0 & 4 & 1 & 1 & 0
\end{bmatrix}
$$

We assume $y$ follows a Poisson distribution for week $t$ with mean $\mu$

$$
y_t \sim \text{Poisson}(\mu_t)
$$

Let $\theta$ be the unknown parameters

$$
\theta^{\mathrm{T}} =
\begin{bmatrix}
\alpha & \psi_1 & \psi_2 & \psi_3 & \psi_4
\end{bmatrix}
$$

**Log link**

We translate $y_t$ to ensure $\mu_t$ is strictly positive, then apply a log transform.

$$
\log(\mu_t) = \alpha + \sum_{i=1}^{4}\psi_i\log(y_{t-i}+1)
$$

**Priors**

$$
\mu_{\text{prior}}^{\mathrm{T}} =
\begin{bmatrix}
\mu_\alpha & \mu_{\rho 1} & \mu_{\rho 2} & \mu_{\rho 3} & 0
\end{bmatrix}
$$

$$
\Sigma_{\text{prior}}^{\mathrm{T}} = \text{diag}(4,1,1,1,1)
$$

$$
\{\alpha,\psi_1,\psi_2,\psi_3,\psi_4\} \sim \text{MultivariateNormal}(\mu_{\text{prior}},\Sigma_{\text{prior}})
$$

**Initial priors**

$$
\alpha \sim \text{Normal}(\mu = 0, \sigma_\alpha^2 = 4)
$$

$$
\psi_1,\psi_2,\psi_3,\psi_4, \sim \text{Normal}(\mu = 0, \sigma_\rho^2 = 1)
$$

**Density**

$$
P(\theta) = \frac{1}{\sqrt{(2\pi)^5 \begin{vmatrix} \Sigma_{\text{prior}} \end{vmatrix}}}
\exp\left(-\frac{1}{2}(\theta - \mu_{\text{prior}})^{\mathrm{T}}\Sigma_{\text{prior}}^{-1}(\theta - \mu_{\text{prior}})\right)
$$

**Likelihood**

$$
L(\theta) = \prod_{t=5}^{n}\frac{\mu_t^{y_t}e^{-\mu_t}}{y_t!}
$$

Therefore the log likelihood $\ell(\theta)$ is give by

$$
\ell(\theta) = \log\left(\prod_{t=5}^{n}\frac{\mu_t^{y_t}e^{-\mu_t}}{y_t!}\right) = \sum_{t=5}^{n}y_t\log(\mu_t) - \mu_t - \log(y_t!)
$$

**Posterior**

$$
\log(P(\theta)) = \sum_{t=5}^{n}(y_t\log(\mu_t)-\mu_t - \log(y_t!)) - \frac{1}{2}(\theta - \mu_{\text{prior}})^{\mathrm{T}}\Sigma_{\text{prior}}^{-1}(\theta-\mu_{\text{prior}}) + c
$$

for some constant $c$.

**Markov Chain Monte Carlo**

**Partial Autocorrelation Function**

Let $\hat{y}_{t+k}$ and $\hat{y}_t$ be linear combinations of

$$
\{y_{t+1},y_{t+2},\dots,y_{t+k-1}\}
$$

The partial autocorrelation of lag $k$ is given by

$$
\phi_{k,k} = \text{corr}(y_{t+k}-\hat{y}_{t+k}, y_t - \hat{y}_t) \quad \text{for} \ k \ge 2
$$

**Durbin-Levinson Algorithm**

$$
\phi_{n,k} = \phi_{n-1,k} - \phi_{n,n}\phi_{n-1,n-k}, \quad k = 1,2,3,4
$$

Let $\phi_{1,1} = \tanh(\psi_1)$. We define $\rho_i = \phi_{4,i}$

$$
\phi_{1,1} = \tanh(\psi_1)
$$

$$
\phi_{2,2} = \tanh(\psi_2), \quad \phi_{2,1} = \phi_{1,1} - \phi_{2,2}\phi_{1,1}
$$

$$
\phi_{3,3} = \tanh(\psi_3), \quad \phi_{3,1} = \phi_{2,1} - \phi_{3,3}\phi_{2,2}, \quad \phi_{3,2} = \phi_{2,2} - \phi_{3,3}\phi_{2,1}
$$

$$
\phi_{4,4} = \tanh(\psi_4), \quad \phi_{4,1} = \phi_{3,1} - \phi_{4,4}\phi_{3,3}, \quad \phi_{4,2} = \phi_{3,2} - \phi_{4,4}\phi_{3,2}, \quad \phi_{4,3} = \phi_{3,3} - \phi_{4,4}\phi_{3,1}
$$
