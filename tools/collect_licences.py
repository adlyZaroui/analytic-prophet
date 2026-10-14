"""Put the third-party licences where a built wheel will carry them.

Issue #97. A binary wheel contains code compiled from Eigen (MPL-2.0) and
LBFGSpp (MIT), so it has to carry their terms. Neither is vendored -- both are
fetched at build time by tools/fetch_headers.sh -- which means nothing in the
repository holds their licence text and the wheel would ship without it.

This copies both out of the fetched trees into `analytic_prophet/licences/`,
which `package-data` ships. Run by cibuildwheel's `before-all`, after the
fetch.

Prophet's licence joins them (#199), and differs in one respect: it is
committed rather than fetched. The Python here ports Prophet's -- every `[fc]`
in the package marks a place that follows it -- so the derived code lives in
this repository and so does its notice, as `LICENSE-PROPHET` at the root.
setuptools' default licence globs (`LICEN[CS]E*`) already put that file in
every wheel's metadata, including one built from the sdist on a platform with
no wheel; this copies it beside the other two so `licences/` holds every
third party's terms and `NOTICE` can say which files they cover.

On MPL-2.0: it is file-level copyleft, and the obligation it creates for a
binary is to make the source form of the covered files available. Eigen is
published unmodified at the pinned version in the URL recorded here, and
nothing in this project modifies it, which is what NOTICE records.
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DESTINATION = HERE / "analytic_prophet" / "licences"

SOURCES = {
    "LICENSE.eigen": ("eigen", ("COPYING.MPL2", "COPYING.README", "LICENSE")),
    # LBFGSpp names it LICENSE.md; the tuple is ordered candidates rather
    # than one name because upstreams rename these.
    "LICENSE.lbfgspp": ("lbfgspp", ("LICENSE.md", "LICENSE", "COPYING")),
}

# Committed, not fetched: see the module docstring.
COMMITTED = {"LICENSE.prophet": HERE / "LICENSE-PROPHET"}

# The modules that port Prophet's Python. tests/test_wheels.py holds this to the
# package: every module with an `[fc]` marker is named here, except where the
# marker reads "[fc] nothing" -- the convention for "Prophet has no
# counterpart", as in build.py.
DERIVED_FROM_PROPHET = ("__init__.py", "constants.py", "forecaster.py",
                        "make_holidays.py", "models.py", "optimizer.py",
                        "seasonality.py", "serialize.py", "trend.py")

NOTICE = """\
The compiled core in this distribution is built from analytic_prophet/optimize.cpp
against two header-only libraries, which are therefore compiled into it:

  Eigen {eigen_version} -- MPL-2.0
    https://gitlab.com/libeigen/eigen/-/archive/{eigen_version}/eigen-{eigen_version}.tar.gz
    Used unmodified. MPL-2.0 is file-level copyleft: the source form of the
    covered files is the archive above, published by the Eigen project.

  LBFGSpp {lbfgspp_version} -- MIT
    https://github.com/yixuan/LBFGSpp/archive/refs/tags/{lbfgspp_version}.tar.gz
    Used unmodified.

Neither is vendored in the source repository; both are fetched at build time by
tools/fetch_headers.sh at the versions above. Their licence texts are beside
this file.

Parts of the Python in this distribution are derived from Prophet
(https://github.com/facebook/prophet), MIT, Copyright (c) Facebook, Inc. and
its affiliates. Its licence is LICENSE.prophet, beside this file. The modules
that port Prophet's code, marked [fc] where they follow it:

  {derived}

analytic-prophet itself is licensed separately -- see the LICENSE in the
distribution metadata.
"""


def main():
    headers = Path(os.environ.get("AP_HEADERS", "/tmp/ap-headers"))
    if not headers.is_dir():
        print(f"{headers} does not exist; run tools/fetch_headers.sh first",
              file=sys.stderr)
        return 1

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for target, (project, candidates) in SOURCES.items():
        for candidate in candidates:
            source = headers / project / candidate
            if source.exists():
                shutil.copyfile(source, DESTINATION / target)
                print(f"  {target} <- {source}")
                break
        else:
            print(f"no licence file found for {project} in {headers / project}",
                  file=sys.stderr)
            return 1

    for target, source in COMMITTED.items():
        shutil.copyfile(source, DESTINATION / target)
        print(f"  {target} <- {source.relative_to(HERE)}")

    (DESTINATION / "NOTICE").write_text(NOTICE.format(
        eigen_version=os.environ.get("EIGEN_VERSION", "3.4.0"),
        lbfgspp_version=os.environ.get("LBFGSPP_VERSION", "v0.3.0"),
        derived=", ".join(f"analytic_prophet/{name}" for name in DERIVED_FROM_PROPHET)))
    print(f"  NOTICE written to {DESTINATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
