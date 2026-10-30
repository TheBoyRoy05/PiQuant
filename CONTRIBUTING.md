# Contributing

These rules apply to people and to coding agents alike. If an agent does the work, the person running it owns the result.

## Every change goes through a reviewed pull request

- `main` is protected: no direct pushes, no force pushes, no deletion
- Work on a branch named `<type>/<change>`, using the same type as the pull request title, e.g. `feat/roofline-model` or `fix/eval-seed`
  - Types: `feat`, `fix`, `docs`, `test`, `chore`, and `report` for changes under `report/`
- Open a pull request into `main` and get at least one approval from another teammate
  - GitHub does not let you approve your own pull request
  - A new push after approval needs a fresh approval
- Merge with Squash and merge, the only method enabled, so each pull request lands on `main` as one commit
  - The pull request title becomes the commit title, so write it as one: `feat: add pi0.5 roofline model`
  - The pull request description becomes the commit body: say what changed and why
- GitHub deletes the branch after merge

## Before asking for review

- The README still matches how to set up and run the code
- No checkpoints, datasets, rollout videos or other large outputs are committed

## Override

Repository admins can bypass the rule for an emergency, such as a broken `main` blocking everyone before a deadline.

- Say why in the commit message and tell the team
- Agents never use the override

## For coding agents

- Push your branch and open the pull request, then stop and wait for a human review
- Never merge your own pull request
