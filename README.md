# Bayesian Habit Tracking
Autoregressive Poisson model with examples

**Premise**

Each week I count how many instances of bad habits I engage in. These instances are defined by units, such as consuming 60 milligrams of caffeine.

**4-week AR(4) model**

Let $y_t \in \mathbb{Z}_{\ge 0}^5$ be instances of 5 different bad habits for week $t$. For a 4-week model, $p=4$.

$$
y_t^{\mathrm{T}} =
\begin{bmatrix}
7 & 6 & 0 & 4 & 0
\end{bmatrix}
$$

Let $\Phi_{a,b}$ be the lag coefficients $\phi$ between habit $a$ and habit $b$ for each week.

$$
\Phi_{a,b} =
\begin{bmatrix}
\phi_{1,ab} & \phi_{2,ab} & \phi_{3,ab} & \phi_{4,ab}
\end{bmatrix}
$$

Let $\theta_j$ be the 21 unknown parameters for a given habit $j$.

$$
\theta_j^{\mathrm{T}} =
\begin{bmatrix}
\alpha_j & \Phi_{j,j} & \Phi_{j,k_1} & \Phi_{j,k_2} & \Phi_{j,k_3} & \Phi_{j,k_4}
\end{bmatrix}
, \quad
\\{k_1,k_2,k_3,k_4\\} = \\{1,\dots,5\\} \setminus \\{j\\}
$$

We assume $y_{t,j}$ follows a Poisson distribution for week $t$ and habit $j$ with mean $\lambda_{t,j}$

$$
y_{t,j} \sim \text{Poisson}(\lambda_{t,j})
$$

**Priors**

$$
\lambda_{\text{prior}} = \mathbf{0}_{21}
$$

**Hyperprior**

$$
\gamma_j \sim \text{Half-Cauchy}(0,s)
$$

$$
p(\gamma_j) = \frac{2}{\pi s\big(1 + (\gamma_j/s)^2\big)}, \quad \gamma_j > 0
$$

$$
\log\left(\frac{2}{\pi s\big(1 + (\gamma_j/s)^2\big)}\right) = \log\left(\frac{2}{\pi s}\right) - \log\left(1 + \frac{\gamma_j^2}{s^2}\right)
$$

**Distributions**

$$
\alpha_j \sim \text{Normal}(0,4), \quad \Phi_{j,j} \sim \text{Normal}(0,1)
$$

$$
\Phi_{j,k_1},\dots,\Phi_{j,k_4} \mid \gamma_j \sim \text{Normal}\big(0,\gamma_j^2\big), \quad \theta_j \mid \gamma_j \sim \text{MultivariateNormal}\big(\mathbf{0}_{21},\Sigma_{\text{prior}}(\gamma_j)\big)
$$

**Covariance**

$$
\Sigma_{\text{prior}}(\gamma_j) = \text{diag}\big(4,\underbrace{1,1,1,1}_4,\underbrace{\gamma_j^2,\dots,\gamma_j^2}_{16}\big), \quad \Sigma_{\text{prior}}(\gamma_j)^{-1} = \text{diag}\big(0.25,\underbrace{1,1,1,1}_4,\underbrace{\gamma_j^{-2},\dots,\gamma_j^{-2}}_{16}\big)
$$

$$
\begin{vmatrix}
\Sigma_{\text{prior}}(\gamma_j)
\end{vmatrix}
= 4\gamma_j^{32}
$$

**Density**

$$
P(\theta_j \mid \gamma_j) = \frac{1}{\sqrt{(2\pi)^{21}
\begin{vmatrix}
\Sigma_{\text{prior}}(\gamma_j)
\end{vmatrix}}}
\exp\left(-\frac{1}{2}\theta_j^{\text{T}}\Sigma_{\text{prior}}(\gamma_j)^{-1}\theta_j\right)
$$

**Log link**

We translate $y_t$ to ensure $\lambda_t$ is strictly positive, then apply a log transform.

$$
\log(\lambda_{t,j}) = \alpha_j + \sum_{i=1}^{4}\phi_{i,jj}\log(y_{t-i,j}+1) + \sum_{k \ne j}\sum_{i=1}^{4}\phi_{i,jk}\log(y_{t-i,k}+1)
$$

**Log Likelihood**

The likelihood function $L(\theta_j)$ is given by

$$
L(\theta_j) = \prod_{t=5}^{n}\frac{\lambda_{t,j}^{y_{t,j}}e^{-\lambda_{t,j}}}{y_{t,j}!}
$$

Therefore the log likelihood $\ell(\theta_j)$ is given by

$$
\ell(\theta_j) = \log\left(\prod_{t=5}^{n}\frac{\lambda_{t,j}^{y_{t,j}}e^{-\lambda_{t,j}}}{y_{t,j}!}\right) = \sum_{t=5}^{n}\big(y_{t,j}\log(\lambda_{t,j}) - \lambda_{t,j} - \log(y_{t,j}!)\big)
$$

**Log Posterior**

$$
\log\big(P(\theta_j,\gamma_j) \mid y\big) = \ell(\theta_j) - \frac{1}{2}(\theta_j - \lambda_{\text{prior}})^{\mathrm{T}}\Sigma_{\text{prior}}(\gamma_j)^{-1}(\theta_j - \lambda_{\text{prior}}) - \frac{1}{2}\log\big(
\begin{vmatrix}
\Sigma_{\text{prior}}(\gamma_j)
\end{vmatrix}
\big) + \log\big(p(\gamma_j)\big) + c
$$

$$
= \ell(\theta_j) - \frac{1}{2}\theta_j^{\mathrm{T}}\Sigma_{\text{prior}}(\gamma_j)^{-1}\theta_j - \frac{1}{2}\log\big(4\gamma_j^{32}\big) + \log\left(\frac{2}{\pi s\big(1 + (\gamma_j/s)^2\big)}\right) + c
$$

$$
= \ell(\theta_j) - \frac{\alpha_j^2}{8} - \frac{1}{2}\sum_{i=1}^4\phi_{i,jj}^2 - \frac{1}{2\gamma_j^2}\sum_{k \ne j}\sum_{i=1}^4\phi_{i,jk}^2 - \log\left(\gamma_j^{16}\left(1 + \frac{\gamma_j^2}{s^2}\right)\right) + c
$$

for some constant $c$.
