## What

<!-- One or two sentences. Title must follow Conventional Commits, e.g. feat(ai): add PPO trainer -->

## Why

<!-- Link the task/issue (e.g. TASK-301, #12). -->

## Module(s) touched

- [ ] simulation  - [ ] ai  - [ ] backend  - [ ] dashboard  - [ ] edge  - [ ] hardware  - [ ] analysis  - [ ] shared/contracts  - [ ] docs / repo

If you touched another member's folder, explain why and make sure they are a reviewer:

## How was it tested?

<!-- Commands run and results. For SUMO runs: scenario, seeds, metrics. -->

## Checklist

- [ ] Targets `develop`, branch named `<type>/<module>-<short-desc>`
- [ ] CI is green (lint, unit tests, contracts)
- [ ] No secrets, `.env` files, data, model weights, SUMO outputs or files > 2 MB
- [ ] New config keys added to `.env.example`
- [ ] Signal safety limits respected where relevant (min green 10 s, max green 60 s, yellow 3 s, all-red 1 s)
- [ ] Contract changes (`shared/contracts/`) approved by every affected owner and still marked DRAFT where applicable
- [ ] Assumptions marked `ASSUMPTION`; docs/README updated
