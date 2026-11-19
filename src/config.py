"""Experiment parameters for the MINGLE replication.

Every module-level constant documents its meaning, unit, and the paper
reference it is taken from. Where the paper (Chehab, Iacovides, Yazdanparast
and Mandic, "Beyond Co-Movement: Locality by Exposures Enables a Joint
Factor-Graph Framework for Portfolio Diversification", ICAIF'26) does not
specify a value, this is stated explicitly.
"""

# ---------------------------------------------------------------------------
# MINGLE model hyperparameters (Sec. 3.1, Eq. 5-8, Table 1)
# ---------------------------------------------------------------------------
K_FACTORS = 6          # number of latent factors, k (Sec. 4, "MINGLE configuration")
LAMBDA_SMOOTH = 0.1    # exposure-smoothness weight, lambda (Sec. 4)
DELTA_DECORR = 1000.0  # factor decorrelation weight, delta (Sec. 4)
ALPHA_SPARSITY = 2.2   # graph L1 sparsity weight, alpha (Sec. 4)
BETA_DEGREE = 3.0      # degree log-barrier weight, beta (Sec. 4)
OMEGA_DECAY = 0.997    # exponential time-decay base, omega; one-year half-life
                       # of ~252 trading days (Sec. 4)
EPS_DEGREE = 1e-6      # degree log-barrier offset, epsilon; not specified in
                       # the paper, set small enough not to bias d upward

# ADMM penalty parameters rho_B, rho_W, rho_d: not specified in the paper
# (Sec. 3.2, "Implementation notes"). Standard ADMM practice (Boyd et al.,
# ref. [6]) starts at rho = 1 for each consensus constraint.
RHO_B = 1.0
RHO_W = 1.0
RHO_D = 1.0

LR_F = 1e-3            # learning rate for the F gradient step; not specified
                       # in the paper, chosen small enough for stable descent
ADMM_MAX_ITER = 100    # maximum ADMM iterations; not specified in the paper
ADMM_TOL = 1e-4        # convergence tolerance on primal residuals; not
                       # specified in the paper
F_INNER_STEPS = 5      # number of gradient substeps per ADMM iteration for
                       # the F update; not specified in the paper
N_RESTARTS = 1         # number of random ADMM initialisations from which the
                       # lowest-objective solution is kept (Remark 1); the
                       # paper does not specify a restart count; kept at 1 for
                       # the executed configuration to control runtime

# ---------------------------------------------------------------------------
# Adaptive CutV spectral partitioning (reference [11], not detailed in the
# paper beyond "Laplacian eigenspectrum with an information-theoretic
# criterion" and "recursively bipartitions the graph")
# ---------------------------------------------------------------------------
N_CLUSTERS = 24                 # number of leaf clusters, C (Sec. 5)
CUTV_CANDIDATE_EIGENVECTORS = 5  # number of smallest non-trivial eigenvectors
                                 # examined by the information-theoretic
                                 # selection criterion at each cut; not
                                 # specified in the paper

# ---------------------------------------------------------------------------
# Data universe (Sec. 4, "Dataset")
# ---------------------------------------------------------------------------
# Paper universe: 100 most actively traded constituents of 2017 from each of
# the S&P 500, Nikkei 225 and STOXX 600 indices (300 equities total).
UNIVERSE_SIZE_PAPER = 100   # per group, paper configuration
UNIVERSE_SIZE_RUN = 30      # per group, executed configuration (see
                            # results/RESULTS.md, "Deviations from the
                            # paper", for the download-time justification)

LIQUIDITY_RANKING_YEAR = 2017  # year over which "most actively traded" is
                                # measured (Sec. 4, "Dataset")

# ---------------------------------------------------------------------------
# Backtest configuration (Sec. 4-5)
# ---------------------------------------------------------------------------
LOOKBACK_TRADING_DAYS = 504   # 2-year lookback window, T (Sec. 4, "Dataset":
                               # "2-year lookback window (T ~ 504 trading days)")
BACKTEST_START = "2019-01-01"  # paper backtest start (Sec. 4/5)
BACKTEST_END_PAPER = "2026-03-31"   # paper backtest end (Sec. 5, "Jan
                                     # 2019-March 2026 backtest period")
BACKTEST_END_RUN = "2026-03-31"     # executed backtest end; identical unless
                                     # realised data coverage is shorter (see
                                     # results/RESULTS.md)
SAMPLE_DOWNLOAD_START = "2016-10-01"  # earliest date downloaded, giving a
                                       # full 2-year lookback before the first
                                       # 2019-01 rebalance
TRANSACTION_COST_BPS = (0, 5, 10, 20, 50)  # one-way turnover cost levels, bps
                                            # (Table 2)
ANNUALISATION_DAYS = 252       # trading days per year for CAGR/Sharpe/Calmar;
                                # standard convention, not explicitly restated
                                # in the paper's text
RISK_FREE_RATE = 0.0           # annual risk-free rate for excess returns; not
                                # specified in the paper

# VIX-based regime thresholds (Remark 2)
VIX_CALM_MAX = 20.0       # Calm: VIX < 20
VIX_ELEVATED_MAX = 30.0   # Elevated: 20 <= VIX <= 30; Crisis: VIX > 30

# ---------------------------------------------------------------------------
# Experiment 4: synthetic noise ablation (Fig. 7)
# ---------------------------------------------------------------------------
ABLATION_SNR_DB = (40.0, 20.0, 10.0)   # signal-to-noise ratios, dB (Fig. 7)
ABLATION_N_SEEDS_PAPER = 100           # paper repetition count per SNR level
ABLATION_N_SEEDS_RUN = 20              # executed repetition count (see
                                        # results/RESULTS.md for the runtime
                                        # justification); use --full for 100
ABLATION_TTEST_ALPHA = 0.001           # significance threshold quoted in the
                                        # paper ("p < 0.001")

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
RANDOM_SEED = 0   # not specified in the paper; fixed for reproducibility
