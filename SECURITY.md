# Security

This is a forecasting library. It does not open sockets, read credentials, or execute
anything a caller did not ask it to — with one exception worth knowing about.

## It compiles C++ on your machine

`fit(df)` runs a compiled core, and on first use it builds that core by invoking the system
C++ compiler on `analytic_prophet/optimize.cpp`, which ships with the package. The result is
cached under `~/.cache/analytic-prophet/build` (or `$ANALYTIC_PROPHET_CACHE`), keyed by a
digest of the source and the compile command.

That means installing this package and calling `fit` runs a compiler. The source compiled is
the one in the installed package and nothing else: no download happens, and the compile
command is fixed in `analytic_prophet/build.py`. If you would rather it did not,
`fit(df, backend="python")` is a pure-Python path that compiles nothing.

The cache directory is written to with the invoking user's permissions, and the build goes
to a private staging name and is atomically renamed, so two processes cannot read a
half-written artefact.

## The evaluation suite downloads data

`evaluation/corpora.py` fetches the M4 competition dataset over HTTPS from its public GitHub
repository, on demand, into the same cache root. Nothing in `analytic_prophet/` does this —
it is the evaluation suite only, and it is never reached by fitting or predicting.

## Reporting something

Open an issue, or if you would rather not do so publicly, use GitHub's **Report a
vulnerability** button on the Security tab. This is a research project with no release
cadence to promise against, so the honest answer on timelines is that a report will be
looked at and fixed when it is understood, not within a stated window.
