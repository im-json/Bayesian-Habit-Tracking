# Bayesian-Habit-Tracking

\title{Bayesian Habit Tracking}
\author{Autoregressive Poisson model with examples}
\date{}

\begin{document}

\maketitle

\section*{Premise}

Each week I count how many instances of bad habits I engage in. These instances are defined by units, such as consuming 60 milligrams of caffeine.

\section*{4-week AR(4) model}

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
\alpha & \rho_1 & \rho_2 & \rho_3 & \rho_4
\end{bmatrix}
$$

\subsection*{Log link}

We translate $y_t$ to ensure $\mu_t$ is strictly positive, then apply a log transform.

$$
\log(\mu_t) = \alpha + \sum_{i=1}^{4}\rho_i\log(y_{t-i}+1)
$$

\subsection*{Initial priors}

$$
\alpha \sim \text{Normal}(\mu = 0, \sigma_\alpha^2 = 4) \\
\rho_1,\rho_2,\rho_3,\rho_4, \sim \text{Normal}(\mu = 0, \sigma_\rho^2 = 1) 
$$

$$
\mu_{\text{prior}}^{\mathrm{T}} =
\begin{bmatrix}
\mu_\alpha & \mu_{\rho 1} & \mu_{\rho 2} & \mu_{\rho 3} & 0
\end{bmatrix}
, \quad
\Sigma_{\text{prior}}^{\mathrm{T}} =
\begin{bmatrix}
\text{Var}(\alpha) & \text{Cov}(\alpha,\rho_1) & \text{Cov}(\alpha,\rho_2) & \text{Cov}(\alpha,\rho_3) & 0 \\
\text{Cov}(\rho_1,\alpha) & \text{Var}(\rho_1) & \text{Cov}(\rho_1,\rho_2) & \text{Cov}(\rho_1,\rho_3) & 0 \\
\text{Cov}(\rho_2,\alpha) & \text{Cov}(\rho_2,\rho_1) & \text{Var}(\rho_2) & \text{Cov}(\rho_2,\rho_3) & 0 \\
\text{Cov}(\rho_3,\alpha) & \text{Cov}(\rho_3,\rho_1) & \text{Cov}(\rho_3,\rho_2) & \text{Var}(\rho_3) & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

$$
\{\alpha,\rho_1,\rho_2,\rho_3,\rho_4\} \sim \text{MultivariateNormal}(\mu_{\text{prior}},\Sigma_{\text{prior}})
$$

\subsection*{Density}

$$
P(\theta) = \frac{1}{\sqrt{(2\pi)^5 \begin{vmatrix} \Sigma_{\text{prior}} \end{vmatrix}}}
\exp\left(-\frac{1}{2}(\theta - \mu_{\text{prior}})^{\mathrm{T}}\Sigma_{\text{prior}}^{-1}(\theta - \mu_{\text{prior}})\right)
$$

\subsection*{Likelihood}

$$
L(\theta) = \prod_{t=5}^{n}\frac{\mu_t^{y_t}e^{-\mu_t}}{y_t!}
$$

Therefore the log likelihood $\ell(\theta)$ is give by

$$
\ell(\theta) = \log\left(\prod_{t=5}^{n}\frac{\mu_t^{y_t}e^{-\mu_t}}{y_t!}\right) = \sum_{t=5}^{n}y_t\log(\mu_t) - \mu_t - \log_(y_t)!
$$

\subsection*{Posterior}

$$
\log(P(\theta)) = \sum_{t=5}^{n}(y_t\log(\mu_t)-\mu_t - \log(y_t!)) - \frac{1}{2}(\theta - \mu_{prior})^{\mathrm{T}}\Sigma_{prior}^{-1}(\theta-\mu_{prior}) + c
$$

\section*{Markov Chain Monte Carlo}

\subsection*{Partial Autocorrelation Function}

Let $\hat{y}_{t+k}$ and $\hat{y}_t$ be linear combinations of

$$
\{y_{t+1},y_{t+2},\dots,y_{t+k-1}\}
$$

The partial autocorrelation of lag $k$ is given by

$$
\phi_{k,k} = \text{corr}(y_{t+k}-\hat{y}_{t+k}, y_t - \hat{y}_t) \quad \text{for} \ k \ge 2
$$

Similarly,

$$
\phi_{n,k} = \phi_{n-1,k} - \phi_{n,n}\phi_{n-1,n-k} \quad \text{for} \ 1 \le k \le n - 1
$$

\subsection*{Durbin-Levinson Algorithm}

The partial autocorrelation of $y_t$ is calculated by

$$
\phi_{n,n} = \frac{f(n)-\sum_{k-1}^{n-1}\phi_{n-1,k}f(n-k)}{1-\sum_{k=1}^{n-1}\phi_{n-1,k}f(k)}
$$

\end{document}
