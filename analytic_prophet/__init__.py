"""Facebook Prophet's fitting engine, with a hand-derived analytic gradient.

The model itself lives in `forecaster.py`, mirroring Prophet's own layout --
`prophet/forecaster.py` holds its `Prophet` class the same way. `optimize.cpp`
beside it is the compiled core, which is this package's answer to
`prophet/models.py`: where Prophet hands the problem to Stan, this hands it to
a gradient written out by hand.

    from analytic_prophet import AnalyticProphet

    model = AnalyticProphet()
    model.fit_cpp(df)
    forecast = model.predict(model.make_future_dataframe(periods=30))

The star import is deliberate. This package is a facade over a single module,
and enumerating its surface here would mean maintaining the same list twice --
with a missing name failing at import time in whatever used it. Anything
reaching for a module *global* rather than a value, which in practice means a
test monkeypatching `minimize` or `load_cpp_module`, must import
`analytic_prophet.forecaster` and patch it there: rebinding a name on this
facade leaves the module's own global untouched, and the patch silently does
nothing.
"""
from .forecaster import *          # noqa: F401,F403
from .forecaster import AnalyticProphet   # noqa: F401  -- the one that matters
