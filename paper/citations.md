# What each citation is used to say, and the passage it rests on

Every source the paper cites was read for **what the paper uses it to say**, not only
for its metadata (#177). This file is the record, so a reviewer can check a claim
without re-reading the source. It applies the same discipline #138 applied to the
upstream issues. Each DOI was resolved against Crossref, and each quotation below
appears verbatim in its source. `tests/test_paper_citations.py` holds the
bibliography, the citations and every quotation in the `.tex` files to this file.

Two claims the first draft made did not survive this check, and the text was
corrected:

- **"Stan documents its optimizers as methods for smooth objectives."** Stan's
  optimization chapter does not say that. It points to Nocedal & Wright for
  "description and analysis". The smoothness is in Nocedal & Wright, and Stan's own
  warning about absolute values is in its *functions* reference, not on the
  distribution Prophet uses. The paper now says exactly that, and the thesis reads
  "solvers designed for smooth objectives".
- **"That quasi-Newton methods struggle on ℓ1-penalised objectives is textbook
  [Nocedal & Wright; Lewis & Overton]."** Nocedal & Wright omit nonsmooth
  optimization rather than discuss it. Lewis & Overton find BFGS *works* on nonsmooth
  functions, given the right line search. The statement is now attributed to Andrew &
  Gao, who make it, and Lewis & Overton are cited for what they actually show.

Re-checking the upstream threads also corrected two attributions in §6, recorded
under their entries below.

---

## Prophet and Stan

**`taylor2018forecasting`** — Taylor & Letham, *The American Statistician* 72(1):37–45,
2018. Checked against the preprint hosted by Prophet
(`facebook.github.io/prophet/static/prophet_paper_20170113.pdf`).

- the changepoint prior: "use the prior δj ∼ Laplace(0, τ)", introduced as "a sparse
  prior on δ" for automatic changepoint selection;
- the fit: "For model fitting we use Stan's L-BFGS to find a maximum a posteriori
  estimate."

**`carpenter2017stan`** — Carpenter et al., *J. Stat. Softw.* 76(1), 2017.

- "The default optimizer in Stan is the limited-memory Broyden-Fletcher-Goldfarb-Shanno
  (L-BFGS) optimizer (Nocedal and Wright 2006)."
- the stopping rule: "The optimizer terminates when any of the log density, gradient,
  or parameter values are within their specified tolerance."

**`stan2025reference`** — Stan Reference Manual, version 2.37, the CmdStan version
Prophet bundles (released 2025-09-02). Checked at
`mc-stan.org/docs/2_37/reference-manual/optimization.html`.

- "Stan provides three different optimizers, a Newton optimizer, and two related
  quasi-Newton algorithms, BFGS and L-BFGS; see Nocedal and Wright (2006) for thorough
  description and analysis of all of these algorithms."
- The chapter contains no statement about smoothness, differentiability or
  non-smooth objectives. That absence is why the paper does not say Stan
  "documents" its optimizers as smooth methods.

**`stan2025functions`** — Stan Functions Reference, version 2.37. Checked at
`mc-stan.org/docs/2_37/functions-reference/real-valued_basic_functions.html` and
`.../unbounded_continuous_distributions.html`.

- Under "Step-like functions", whose subsection "Absolute value functions" holds
  `abs`: "These functions can seriously hinder sampling and optimization efficiency
  for gradient-based methods (e.g., NUTS, HMC, BFGS) if applied to parameters".
- The "Double exponential (Laplace) distribution" section gives the density
  (1/2σ) exp(−|y − μ|/σ) and carries no comparable warning.

**Stan source, not in the bibliography** — `stan/optimization/bfgs_linesearch.hpp` in
CmdStan 2.37.0, cited in §3 by path. `WolfeLineSearch` performs "a line search …
satisfying the strong Wolfe conditions", with `c1 = 1e-4` and `c2 = 0.9` in
`bfgs.hpp`. Stan's Hessian for Newton (§4.6) is `grad_hess_log_prob.hpp`: a
four-point finite difference of the gradient with `epsilon = 1e-3`.

## Smooth and nonsmooth optimization

**`nocedal2006numerical`** — Nocedal & Wright, *Numerical Optimization*, 2nd ed.,
Springer, New York, 2006.

- §2.1, "Nonsmooth problems": "This book focuses on smooth functions, by which we
  generally mean functions whose second derivatives exist and are continuous."
- Preface, "Topics not covered": "We omit some important topics, such as network
  optimization, integer programming, stochastic programming, nonsmooth optimization,
  and global optimization."
- Assumption 6.1, global convergence of BFGS: "The objective function f is twice
  continuously differentiable."
- Theorem 3.2 (Zoutendijk), the Wolfe-condition line-search result, assumes f
  "continuously differentiable" with a Lipschitz continuous gradient.

**`lewis2013nonsmooth`** — Lewis & Overton, *Math. Program.* 141:135–163, 2013.
Checked against the authors' copy (`cs.nyu.edu/~overton/papers/pdffiles/bfgs_inexactLS.pdf`).

- With a weak-Wolfe line search, BFGS "consistently converges to local minimizers on
  all but the most difficult class of examples"; "the convergence rate is observed to
  be linear with respect to the number of function evaluations".
- "For nonsmooth optimization, it is clear that enforcing the strong Wolfe condition
  is not possible in general, and it is essential to base the line search on the
  simpler weak Wolfe condition."
- They recommend that L-BFGS and L-BFGS-B "include the weak Wolfe line search …
  as an optional alternative to the strong Wolfe line search that is currently
  implemented". They do not study limited-memory variants themselves.

**`andrew2007scalable`** — Andrew & Gao, ICML 2007, pp. 33–40.

- Abstract: "The l-bfgs limited-memory quasi-Newton method is the algorithm of choice
  for optimizing the parameters of large-scale log-linear models with L2
  regularization, but it cannot be used for an L1-regularized loss due to its
  non-differentiability whenever some parameter is zero." OWL-QN is their remedy.

**`byrd1995limited`** — Byrd, Lu, Nocedal & Zhu, *SIAM J. Sci. Comput.*
16(5):1190–1208, 1995. Cited only as the bound-constrained limited-memory method
L-BFGS-B. Metadata checked against Crossref.

## Methods for the ℓ1 term, and the split

**`friedman2010regularization`** — Friedman, Hastie & Tibshirani, *J. Stat. Softw.*
33(1):1–22, 2010.

- "The algorithms use cyclical coordinate descent, computed along a regularization
  path", for penalties including "ℓ1 (the lasso)".

**`beck2009fast`** — Beck & Teboulle, *SIAM J. Imaging Sci.* 2(1):183–202, 2009.

- "iterative shrinkage-thresholding algorithms (ISTA) … which can be viewed as an
  extension of the classical gradient algorithm"; FISTA is their accelerated variant.
  The paper says no more than that these handle the penalty directly.

**`tibshirani1996regression`** — Tibshirani, *JRSS B* 58(1):267–288, 1996.

- §6 gives the split as the second of two lasso algorithms: "A completely different
  algorithm for this problem was suggested by David Gay. We write each βj as
  βj⁺ − βj⁻, where βj⁺ and βj⁻ are non-negative." … "One can show that this new problem
  has the same solution as the original problem."

**`chen2001atomic`** — Chen, Donoho & Saunders, *SIAM Review* 43(1):129–159, 2001
(the SIGEST reprint). Checked against `web.stanford.edu/group/SOL/papers/BasisPursuit-SIGEST.pdf`.

- §3.1 reformulates basis pursuit as a standard-form LP with "x ⇔ (u; v), α ⇔ u − v".
- "(The equivalence of minimum ℓ1 optimizations with LP has been known since the
  1950s; see [2].)"

**`figueiredo2007gradient`** — Figueiredo, Nowak & Wright, *IEEE J. Sel. Top. Signal
Process.* 1(4):586–597, 2007.

- §II-A: "this is done by splitting the variable x into its positive and negative
  parts. Formally, we introduce vectors u and v and make the substitution x = u−v,
  u ≥ 0, v ≥ 0", with ‖x‖₁ = 1ᵀu + 1ᵀv. The result is a bound-constrained QP, solved
  by gradient projection.

**`efron2004least`** — Efron, Hastie, Johnstone & Tibshirani, *Ann. Statist.*
32(2):407–499, 2004 (with discussion).

- "the LARS modification calculates all possible Lasso estimates for a given
  problem"; the text refers to "the piecewise linear Lasso path".

## Prophet's issue tracker (§6)

Re-read in full for this check: comments, authors and GitHub author associations.

- **#1422** (closed). The opening post: "only the changepoints having a delta > 0.01
  (**threshold**) are selected. The piecewise linear trend model uses these selected
  changepoints as its breakpoints." The reply "Delta is an absolute threshold applied
  only for visualization and doesn't affect the model or its predictions" is by
  `benmwhite`, whose GitHub association with the repository is *none*. No maintainer
  posted. The paper said "a contributor corrects it … which upstream resolved
  correctly"; it now says another participant corrected it and the thread resolved it.
  Prophet's `add_changepoints_to_plot` uses `threshold: float = 0.01` on
  `np.abs(np.nanmean(m.params['delta'], axis=0))`.
- **#1280** (closed), opened by `FredericoCoelhoNunes`: a change of
  `changepoint_prior_scale` "from 0.01 to 0.009999999776482582" moved accuracy by
  "around 2%". `bletham` (Ben Letham, a Prophet author): "One possibility is that in
  the earlier dataset the optimizer was running into issues with one of the
  changepoint prior scales"; "this seems inline with expectation to me" (the source's
  spelling).
- **#842** (closed). Ten weekly points, `seasonality.mode = "multiplicative"`,
  `n.changepoints = 2`, an extra regressor; the fit stalls until the process is
  killed. `bletham`: the model parameters "are not very well specified";
  "overparameterization"; "the next version I will put a rule that defaults to Newton
  if there are less than X datapoints".
- **#1032** (open), opened by `bletham` from humphreyapplebee's report: logistic
  growth, multiplicative seasonality; "With LBGFS … the training stalls"; with Newton,
  "B[1] is -nan, but must not be nan!". The NaN log-probabilities come from `bletham`'s
  reproduction in #842: "for me the L-BFGS doesn't freeze but it does exit with an
  error since it starts getting log probabilities of NaN, and then it automatically
  retries with Newton which also runs into NaNs and throws the B[1] error". The paper
  had attributed that to #1032 itself; it now says where it comes from.
