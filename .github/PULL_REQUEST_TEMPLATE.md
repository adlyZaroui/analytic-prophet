<!--
What this repository's PRs do, and what reviewers look for. Delete what does not
apply -- a one-line typo fix does not need a measurement section.
-->

Closes #

## What was wrong

<!-- The behaviour, not the diff. Where it is a defect, what it did instead. -->

## What this does about it

## Measured

<!--
Numbers, not adjectives. At minimum:
  - the suite:      N tests pass, M new, 0 failures
  - the benchmarks, if anything on the fit or predict path changed
  - any published number that moved, and every document updated with it
-->

## Checklist

- [ ] Branched from `main`, not from another branch
- [ ] `pytest` passes in full
- [ ] The benchmarks were run, if this touches the fit or predict path
- [ ] A test fails without this change and passes with it
- [ ] Anything whose job is to catch drift was mutation-checked
- [ ] Deliberate divergences from Prophet are recorded in `docs/deviations.md`
