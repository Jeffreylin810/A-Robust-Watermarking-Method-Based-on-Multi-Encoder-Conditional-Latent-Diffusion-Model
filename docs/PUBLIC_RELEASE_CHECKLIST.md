# Public Release Checklist

Use this checklist before publishing the public GitHub version.

## Safety

- remove or generalize hard-coded personal local paths where practical
- confirm no tokens, secrets, passwords, or private credentials are committed
- confirm no private email addresses or account-only notes should remain public
- confirm no private lab-only storage paths need to stay hidden

## Legal And Distribution

- keep datasets out of the repository
- keep model files and VAE files out of the repository
- keep checkpoints out of the repository unless their redistribution is clearly
  allowed and intentionally desired
- include `LICENSE`
- include `docs/THIRD_PARTY_AND_LICENSE_NOTES.md`

## Portfolio Presentation

- keep `docs/PORTFOLIO_OVERVIEW.md` accurate
- keep the main story focused on:
  - trade-off design
  - milestone experiments
  - final formal reruns
  - honest near-miss reporting
- avoid drowning the repo front page in every exploratory branch

## Academic Handoff Coverage

- keep `docs/ACADEMIC_HANDOFF_OVERVIEW.md` accurate
- keep `docs/HISTORICAL_MILESTONES.md` aligned with the chosen milestone policy
- keep final formal summaries for `exp52`, `exp53`, and `exp54`
- keep thesis audit and thesis-ready tables
- keep path/setup notes for external resources

## NAS Package Boundary

- minimal NAS package should include SD2.1 / VAE backup and core summaries
- full NAS package may include additional curated final artifacts
- pre-formal checkpoints remain excluded unless the policy changes later

## Final Sanity Check

- open the repo as if you were an interviewer and verify the first reading path
  makes sense
- open the repo as if you were a new student and verify the handoff path makes
  sense
- verify that public and academic narratives do not contradict each other
