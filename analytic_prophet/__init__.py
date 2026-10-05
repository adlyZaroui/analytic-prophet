"""Facebook Prophet's fitting engine, with a hand-derived analytic gradient.

    from analytic_prophet import AnalyticProphet

    model = AnalyticProphet()
    model.fit(df)
    forecast = model.predict(model.make_future_dataframe(periods=30))

Where things are:

    forecaster.py     the model. [fc] prophet/forecaster.py, which holds the
                      `Prophet` class the same way and at a comparable size.
    constants.py      the numbers the model is defined by, [stan]/[fc]-sourced.
    layout.py         where each parameter sits in the flat vector.
    seasonality.py    the Fourier basis, the registry, the selection rule.
    make_holidays.py  [fc] prophet/make_holidays.py, plus the design columns.
    trend.py          the three growth modes and their derivatives.
    optimizer.py      projected Newton, and the tolerances runs stop on.
    models.py         [fc] prophet/models.py -- the compiled backend's loader.
    serialize.py      [fc] prophet/serialize.py -- a fitted model to and from
                      JSON. Reached as `from analytic_prophet.serialize import
                      model_to_json`, and deliberately not re-exported here,
                      because `import prophet` does not expose its one either.
    optimize.cpp      that backend.

The last four have no Prophet counterpart worth the name, and that is the point:
Stan supplies the layout, the derivatives and the optimizer there. Writing them
down is what this project is.

`forecaster.py` imports every name the other modules define, so the star import
below still re-exports the whole surface from one place -- and so a test
patching `forecaster.minimize` patches the name the model actually reads.

**Patch where a name is looked up, not where it is defined.** `load_cpp_module`
reads `CPP_MODULE_NAME` from `models`' globals, so rebinding
`forecaster.CPP_MODULE_NAME` changes a copy nothing consults. That is not
hypothetical -- it is the one test that broke when this package was split, and
it broke loudly only because the assertion was about behaviour rather than
about the patch.
"""
__version__ = "0.1.0"

from .forecaster import *          # noqa: F401,F403,E402
from .forecaster import AnalyticProphet   # noqa: F401,E402  -- the one that matters
from . import (constants, forecaster, layout, make_holidays,  # noqa: F401,E402
               models, optimizer, seasonality, trend)
