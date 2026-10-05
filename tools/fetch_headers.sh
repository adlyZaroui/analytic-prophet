#!/usr/bin/env bash
# Fetch the two header-only libraries the C++ core needs, at pinned versions.
#
# Issue #97. They are not vendored: pinned tarballs keep the repository free of
# third-party source and keep the versions visible in one place, and header-only
# means there is nothing to link and nothing to build.
#
# Run by cibuildwheel's `before-all`, which means inside the manylinux container
# on Linux and on the runner itself on macOS -- so this has to work in both, with
# only curl and tar assumed.
#
# Licences: Eigen is MPL-2.0, LBFGSpp is MIT. Both are copied next to the headers
# and into the wheel by tools/collect_licences.py, because a wheel contains code
# compiled from both and has to carry their terms.
set -euo pipefail

EIGEN_VERSION="${EIGEN_VERSION:-3.4.0}"
LBFGSPP_VERSION="${LBFGSPP_VERSION:-v0.3.0}"
DESTINATION="${AP_HEADERS:-/tmp/ap-headers}"

mkdir -p "$DESTINATION"
cd "$DESTINATION"

if [ ! -d "eigen" ]; then
    echo "fetching Eigen $EIGEN_VERSION"
    curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" \
        -o eigen.tar.gz
    mkdir -p eigen
    tar -xzf eigen.tar.gz -C eigen --strip-components=1
    rm eigen.tar.gz
fi

if [ ! -d "lbfgspp" ]; then
    echo "fetching LBFGSpp $LBFGSPP_VERSION"
    curl -fsSL "https://github.com/yixuan/LBFGSpp/archive/refs/tags/${LBFGSPP_VERSION}.tar.gz" \
        -o lbfgspp.tar.gz
    mkdir -p lbfgspp
    tar -xzf lbfgspp.tar.gz -C lbfgspp --strip-components=1
    rm lbfgspp.tar.gz
fi

# The two files the build actually looks for. Failing here rather than at
# compile time says which download went wrong.
test -f "$DESTINATION/eigen/Eigen/Dense"        || { echo "Eigen/Dense missing"; exit 1; }
test -f "$DESTINATION/lbfgspp/include/LBFGSB.h" || { echo "LBFGSB.h missing"; exit 1; }

echo "headers ready in $DESTINATION"
