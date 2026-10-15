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

**`roualdes2023bridgestan`** — Roualdes, Ward, Carpenter, Seyboldt & Axen, *Journal of
Open Source Software* 8(87):5236, 2023. Metadata checked against Crossref (published
2023-07-22). Read in the published article's JATS record
(`www.theoj.org/joss-papers/joss.05236/10.21105.joss.05236.jats`). Cited in §4.2 for
one thing: Stan's own gradient of a model is reachable from outside Stan, in memory, so
a solver outside Stan does not need a gradient of its own.

- "BridgeStan provides in-memory access to the methods of Stan models through Python,
  Julia, R, and Rust."
- "Using memory allocated within such higher level languages, BridgeStan provides
  computations of the log joint density function and corresponding gradient of a Stan
  model, which is itself implemented using highly templated C++ from the Stan math
  library."

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
`bfgs.hpp`. Stan's Hessian for Newton (§5.7) is `grad_hess_log_prob.hpp`: a
four-point finite difference of the gradient with `epsilon = 1e-3`.

**`carpenter2017lasso`** — Bob Carpenter, comment of 14 February 2017 on "Lasso regression
etc in Stan", posted by Andrew Gelman the same day on *Statistical Modeling, Causal
Inference, and Social Science*. Cited in §1 for what Stan's own community knew. Read in
full, the post and all 13 comments. The live page refuses automated readers, so it was read
from the Internet Archive's snapshots
(`web.archive.org/web/20201125221139/https://statmodeling.stat.columbia.edu/2017/02/14/lasso-regression-etc-stan/`,
and those of 2019-06-24 and 2019-12-05, which carry the same comment).

- The post relays Ben Goodrich's answer to "Someone on the users list" who "asked about
  lasso regression in Stan". Carpenter's comment opens: "I'll also paste in an edited
  version of my reply on the users list". That is why the paper calls it a reply on the
  Stan users' list.
- "Stan provides penalized maximum likelihood. Unlike specialized lasso software such as
  truncated gradient descent, straight-up optimization is unlikely to find estimates of
  exactly zero for parameters—it will only get there from rounding below 10e-300 or so."
- "The absolute value in the L1 penalty (same as in Laplace prior) is bad news in that it
  means the posterior isn't continuously differentiable. Stan's samplers and optimizers can
  run into problems when the parameter jumps across that non-differentiable point."
- Not cited, but part of the thread: Goodrich replies that "In rstanarm, we write the
  Laplace (and "lasso") prior as a scale mixture of normals, so there is no lack of
  differentiability". That is a remedy for sampling, and it adds a latent scale per
  coefficient, so it changes the problem a mode is taken of.

The source sets apostrophes and quotation marks typographically; they are straight here
and in the paper.

**Prophet's documentation, not in the bibliography** — the quick start
(`facebook.github.io/prophet/docs/quick_start.html`, read 2026-10-15) introduces the
series §2 names the Peyton Manning series: "As an example, let's look at a time series
of the log daily page views for the Wikipedia page for Peyton Manning." The paper
describes it in those terms and does not quote it.

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

## Smooth reparametrisations of the ℓ1 term (§8)

Cited in §8 together with `paper/experiments/inside_stan.py` (#202), which runs two of
them in Prophet's own Stan program, on Stan's own L-BFGS: the product form, reported in
§8, and the split declared non-negative, reported in §4.2 as the reason the fix's solver
runs outside Stan (#204). The sources say the
reparametrised problem has the same minimisers. They do not say that a given solver
finds them, and the paper cites them for the first and measures the second.

**`hoff2017lasso`** — Hoff, *Comput. Stat. Data Anal.* 115:186–198, 2017. Metadata
checked against Crossref. Read in full in the arXiv version (arXiv:1611.00040v2,
19 May 2017).

- The parametrisation: "reparametrize the model so that β= u◦v, where "◦" is the
  Hadamard (element-wise) product", with objective g(u,v) = h(u◦v) + λ(u⊤u + v⊤v)/2.
- Lemma 1, for any h: "infβf(β) = infu,vg(u,v)", and "if (û,v̂) is a local minimum of
  g, then β̂ = û◦v̂ is a local minimum of f". For the lasso: "In other words, lasso
  estimates can be obtained from local minimizers of g."
- The stationary point at the origin: "the point (0,0) ∈ Rp×Rp is a stationary point of g
  but is not a local minimum". Prophet starts every rate at zero, so #202's runs must
  choose a starting value for a zero rate; the paper says so.

**`kolb2026smoothing`** — Kolb, Müller, Bischl & Rügamer, *Machine Learning*
115(4):87, 2026 (CC BY 4.0). Metadata and abstract checked against Crossref. Read in
full in the arXiv version (arXiv:2307.03571v4, 8 Apr 2026), whose abstract is the
published one. An earlier arXiv version worded the abstract differently. The paper
quotes the published wording.

- Abstract: "We prove that the surrogate objective is equivalent in the sense of identical
  global and local minima, thereby avoiding the introduction of spurious solutions."
  Theorem 1 states it for the parametrisation and surrogate penalty used here (their
  HPP: β = u⊙v with ‖u‖² + ‖v‖², a surrogate for 2‖β‖₁).
- §7, on critical points: "such a matching property does not necessarily hold for
  critical points of the surrogate", and ξ = 0 is one "irrespective of the gradient".
- §7, on what a gradient method reaches: "the iterates can only asymptotically approach
  theoretically zero values in a finite number of steps." They call this benign. In the
  paper it is the reason no run of #202 produces an exact zero.

**`poon2021smooth`** — Poon & Peyré, *Advances in Neural Information Processing
Systems* 34:1543–1555, 2021. Pages and volume checked against the proceedings' own
BibTeX. NeurIPS assigns no DOI, so the entry carries the arXiv record's DOI. Read in
full in the arXiv version (arXiv:2106.01429v2, 30 Sep 2022).

- Abstract: "Despite being non-convex, we show that there are no spurious minima and that
  saddle points are "ridable", so that there always exists a descent direction. We thus
  advocate for the use of a BFGS quasi-Newton solver".
- These results are for their *bilevel* form. One factor is eliminated by solving its
  inner least-squares problem, and Theorem 3 is for a squared loss. The plain (u, v)
  problem on L-BFGS is one of the solvers they benchmark against ("Non-cvx-LBFGS"). The
  paper cites them for the landscape of a reparametrised lasso, not for Stan's L-BFGS on
  Prophet's model.

**Considered and not cited: Ziyin & Wang (ICML 2023)**, "spred". It applies the same
Hadamard device to train sparse neural networks by stochastic gradient descent. Kolb et
al. review it with the rest of that literature. It adds nothing about the
lasso's minimisers that the three entries above do not, and nothing about quasi-Newton
methods.

## The M4 data (§7, and the census throughout)

**`makridakis2020m4`** — Makridakis, Spiliotis & Assimakopoulos, *Int. J. Forecast.*
36(1):54–74, 2020. Cited for the competition and its 100,000 series. The full text is
behind a paywall and was **not** read for anything beyond the title, abstract and
metadata. The six frequencies and 100,000 series are confirmed instead from the
organizers' info file (next entry), via `paper/results/m4_calendar.csv`: Yearly 23,000,
Quarterly 24,000, Monthly 48,000, Weekly 359, Daily 4,227, Hourly 414.

**`m4methods`** — the organizers' data release, `github.com/Mcompetitions/M4-methods`.
The README says the info file gives each series' "ID (M4id), domain (category),
frequency (Frequency), number of forecasts requested (Horizon), seasonal periods (SP)
and starting date (StartingDate)", and: "Note that originally the starting date wasn't
provided to the participants." No location is recorded.

An earlier draft of §7 said M4's "dates are synthetic". Nothing here supports that
word. What is supported is in `paper/results/m4_calendar.csv` (#178):

- **the dates look partly like placeholders:** 1,506 of the census's 2,714 daily series
  share the timestamp `30-03-01 12:00`, and there are 122 distinct starts among them;
- **this project's loader moves some of them:** `StartingDate` has two-digit years and
  the loader resolves them with a window that moves with the current year, so 43 weekly
  series start in the future when run in 2026.

The section now says those things instead.

## Prophet's issue tracker (§6)

Re-read in full for this check: comments, authors, dates and GitHub author associations.
The paper cites each issue as a link to its thread. None of the four threads mentions
non-differentiability, the kink, L1, the Laplace prior or a subgradient; that was checked
by searching every comment, so §6's "none of them mentions it" stands.

- **[#1422](https://github.com/facebook/prophet/issues/1422)** (closed). The opening post
  reads the plotting cutoff as selecting breakpoints: "only the changepoints having a delta
  > 0.01 (threshold) are selected. The piecewise linear trend model uses these selected
  changepoints as its breakpoints." The source puts **threshold** in bold; the paper quotes
  it plain. The reply "Delta is an absolute threshold applied only for visualization and
  doesn't affect the model or its predictions" is by `benmwhite`, whose GitHub association
  with the repository is *none*. No maintainer posted. The paper said "a contributor
  corrects it … which upstream resolved correctly"; it now says another participant
  corrected it and the thread resolved it. Prophet's `add_changepoints_to_plot` uses
  `threshold: float = 0.01` on `np.abs(np.nanmean(m.params['delta'], axis=0))`.
- **[#1280](https://github.com/facebook/prophet/issues/1280)** (closed). The opening post
  (`FredericoCoelhoNunes`, 2020-01-15) changes `changepoint_prior_scale` "from 0.01 to
  0.009999999776482582", which is exactly `float32(0.01)`, and reports "a decrease of
  around 2% accuracy". Three weeks later the same reporter: "I have not been able to
  reproduce the 2% case", finding "a variation of 0.8% accuracy" on a simplified model.
  `bletham` (Ben Letham, a Prophet author), 2020-02-05: "One possibility is that in the
  earlier dataset the optimizer was running into issues with one of the changepoint prior
  scales; you might try the Newton optimizer which I have found to be a bit more robust
  than the default"; "this seems inline with expectation to me" (the source's spelling).
- **[#842](https://github.com/facebook/prophet/issues/842)** (closed). Ten weekly points,
  `seasonality.mode = "multiplicative"`, `n.changepoints = 2`, an extra regressor; the fit
  stalls until the process is killed. `bletham`, 2019-02-11: the stalls are "happening
  because the model parameters are not very well specified by so little data". 2019-04-22:
  "In the meantime, the next version I will put a rule that defaults to Newton if there are
  less than X datapoints, where X will probably be 100." 2019-05-21: "Pushed to CRAN".
  2019-09-05: "Generally the issue seems to be overparameterization of the model."
- **[#1032](https://github.com/facebook/prophet/issues/1032)** (open), opened by `bletham`
  from humphreyapplebee's report: logistic growth, multiplicative seasonality; "With LBGFS
  … the training stalls"; with Newton, "B[1] is -nan, but must not be nan!". The reporter:
  "training no longer stalls when I modify the number of changepoints (either adding more or
  less) or perturb the training data even very slightly (e.g. by dropping a significant
  figure from one of the dates)". The NaN log-probabilities come from `bletham`'s
  reproduction in #842: "for me the L-BFGS doesn't freeze but it does exit with an error
  since it starts getting log probabilities of NaN, and then it automatically retries with
  Newton which also runs into NaNs and throws the B[1] error". The paper had attributed
  that to #1032 itself; it now says where it comes from.

## Verified for #208 and #212, not yet cited: Prophet's 2023 plan

**`duong2023prophet`** (no bibliography entry until a section cites it) — Cuong Duong,
"facebook/prophet in 2023 and beyond", Medium, 26 February 2023, linked from Prophet's
README as its "2023 Update" (`github.com/facebook/prophet`, README, read 2026-10-14).
Read in full from the Internet Archive's snapshot
(`web.archive.org/web/20230317232722/https://medium.com/@cuongduong_35162/facebook-prophet-in-2023-and-beyond-c5086151c138`),
because Medium refuses automated readers. The canonical URL is
`https://medium.com/@cuongduong_35162/facebook-prophet-in-2023-and-beyond-c5086151c138`.

- The sentence #208 and #212 rest on, in full: "There have been ideas for further
  enhancements to the model, but at this moment we do not plan to make any more changes to
  the underlying Stan model."
- The TL;DR says it more softly: "We do not plan to make large changes to the underlying
  model." A citation must not turn "at this moment" into a permanent policy.
- The post goes on to say: "Instead, we'll focus on ironing out the kinks in existing
  functionality".
