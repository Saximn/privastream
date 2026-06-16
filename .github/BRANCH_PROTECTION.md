# Branch Protection

The `main` branch is the source of truth for PrivaStream and must always be in a
runnable, releasable state. To enforce this, the repository uses a combination
of an automated CI workflow ([`.github/workflows/ci.yml`](./workflows/ci.yml))
and GitHub branch protection rules.

## Recommended settings for `main`

A repository administrator should configure the following branch protection rule
under **Settings → Branches → Branch protection rules** (or via a ruleset) for
the `main` branch:

- **Require a pull request before merging**
  - Require at least **1 approving review**.
  - Dismiss stale pull request approvals when new commits are pushed.
  - Require review from Code Owners (if a `CODEOWNERS` file is added).
- **Require status checks to pass before merging**
  - Require branches to be up to date before merging.
  - Required checks:
    - `Lint (flake8)`
    - `Tests (pytest)`
- **Require conversation resolution before merging.**
- **Require linear history** (prevents messy merge commits).
- **Do not allow bypassing the above settings** (apply rules to administrators).
- **Block force pushes** to `main`.
- **Restrict deletions** of `main`.

These required status checks map directly to the jobs defined in the CI workflow,
so any change that breaks linting or tests cannot be merged into `main`.

## Applying the rule with the GitHub CLI

The same protection can be applied programmatically by an administrator:

```bash
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  /repos/Saximn/privastream/branches/main/protection \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=Lint (flake8)" \
  -f "required_status_checks[contexts][]=Tests (pytest)" \
  -F "enforce_admins=true" \
  -F "required_pull_request_reviews[required_approving_review_count]=1" \
  -F "required_pull_request_reviews[dismiss_stale_reviews]=true" \
  -F "required_linear_history=true" \
  -F "allow_force_pushes=false" \
  -F "allow_deletions=false" \
  -F "restrictions=null"
```

> **Note:** Configuring branch protection requires repository admin permissions
> and cannot be performed from within a pull request. The CI workflow in this
> repository provides the status checks; an administrator must enable the rule
> in the repository settings.
