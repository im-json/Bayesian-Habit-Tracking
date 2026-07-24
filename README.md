# Bayesian-Habit-Tracking

\documentclass{article}
\usepackage[a4paper, total={6.5in, 9.5in}]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{fancyhdr}
\usepackage{graphicx}
\usepackage{mathtools}
\usepackage{amssymb}
\usepackage{multicol}
\usepackage{listings}
\usepackage{upquote}
\usepackage{float}
\usepackage{xcolor}
\usepackage{setspace}
\usepackage{caption}
\usepackage[none]{hyphenat}

\setlength\parindent{0pt}
\newtagform{brackets}{[}{]}
\usetagform{brackets}

\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstset{
    backgroundcolor=\color{backcolour},
    basicstyle=\small\ttfamily\setstretch{1.0},
    columns=fullflexible,
    upquote=true,
    keepspaces=true,
    showstringspaces=false,
    literate={~}{${\sim}$}{1},
    aboveskip=15pt,
    belowskip=15pt
}

\title{Bayesian Habit Tracking}
\author{Autoregressive Poisson model with examples}
\date{}

\begin{document}

\maketitle

\section*{Premise}

Each week I count how many instances of bad habits I engage in. These instances are defined by units, such as consuming 60 milligrams of caffeine.

\section*{4-week AR(4) model}

Let $y$ be instances of bad habits for a given week.

\begin{align*}
y^{\mathrm{T}} =
\begin{bmatrix}
7 & 6 & 0 & 0 & 0 & 0 & 4 & 1 & 1 & 0
\end{bmatrix}
\end{align*}

We assume $y$ follows a Poisson distribution for week $t$ with mean $\mu$

\begin{align*}
y_t \sim \text{Poisson}(\mu_t)
\end{align*}

Let $\theta$ be the unknown parameters

\begin{align*}
\theta^{\mathrm{T}} =
\begin{bmatrix}
\alpha & \rho_1 & \rho_2 & \rho_3 & \rho_4
\end{bmatrix}
\end{align*}

\subsection*{Log link}

We translate $y_t$ to ensure $\mu_t$ is strictly positive, then apply a log transform.

\begin{align*}
\log(\mu_t) = \alpha + \sum_{i=1}^{4}\rho_i\log(y_{t-i}+1)
\end{align*}

\subsection*{Initial priors}

\begin{align*}
\alpha \sim \text{Normal}(\mu = 0, \sigma_\alpha^2 = 4) \\
\rho_1,\rho_2,\rho_3,\rho_4, \sim \text{Normal}(\mu = 0, \sigma_\rho^2 = 1) 
\end{align*}

\begin{align*}
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
\end{align*}

\begin{align*}
\{\alpha,\rho_1,\rho_2,\rho_3,\rho_4\} \sim \text{MultivariateNormal}(\mu_{\text{prior}},\Sigma_{\text{prior}})
\end{align*}

\subsection*{Density}

\begin{align*}
P(\theta) = \frac{1}{\sqrt{(2\pi)^5 \begin{vmatrix} \Sigma_{\text{prior}} \end{vmatrix}}}
\exp\left(-\frac{1}{2}(\theta - \mu_{\text{prior}})^{\mathrm{T}}\Sigma_{\text{prior}}^{-1}(\theta - \mu_{\text{prior}})\right)
\end{align*}

\subsection*{Likelihood}

\begin{align*}
L(\theta) = \prod_{t=5}^{n}\frac{\mu_t^{y_t}e^{-\mu_t}}{y_t!}
\end{align*}

Therefore the log likelihood $\ell(\theta)$ is give by

\begin{align*}
\ell(\theta) = \log\left(\prod_{t=5}^{n}\frac{\mu_t^{y_t}e^{-\mu_t}}{y_t!}\right) = \sum_{t=5}^{n}y_t\log(\mu_t) - \mu_t - \log_(y_t)!
\end{align*}

\subsection*{Posterior}

\begin{align*}
\log(P(\theta)) = \sum_{t=5}^{n}(y_t\log(\mu_t)-\mu_t - \log(y_t!)) - \frac{1}{2}(\theta - \mu_{prior})^{\mathrm{T}}\Sigma_{prior}^{-1}(\theta-\mu_{prior}) + c
\end{align*}

\section*{Markov Chain Monte Carlo}

\subsection*{Partial Autocorrelation Function}

Let $\hat{y}_{t+k}$ and $\hat{y}_t$ be linear combinations of

\begin{align*}
\{y_{t+1},y_{t+2},\dots,y_{t+k-1}\}
\end{align*}

The partial autocorrelation of lag $k$ is given by

\begin{align*}
\phi_{k,k} = \text{corr}(y_{t+k}-\hat{y}_{t+k}, y_t - \hat{y}_t) \quad \text{for} \ k \ge 2
\end{align*}

Similarly,

\begin{align*}
\phi_{n,k} = \phi_{n-1,k} - \phi_{n,n}\phi_{n-1,n-k} \quad \text{for} \ 1 \le k \le n - 1
\end{align*}

\subsection*{Durbin-Levinson Algorithm}

The partial autocorrelation of $y_t$ is calculated by

\begin{align*}
\phi_{n,n} = \frac{f(n)-\sum_{k-1}^{n-1}\phi_{n-1,k}f(n-k)}{1-\sum_{k=1}^{n-1}\phi_{n-1,k}f(k)}
\end{align*}

\end{document}
