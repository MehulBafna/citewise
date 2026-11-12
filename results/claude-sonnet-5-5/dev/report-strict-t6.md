# PubMedQA evaluation

Generated 2026-11-12T11:44:15+00:00. Split: dev, 50 questions (seed 0). Main model: `claude-sonnet-5-5`, fast model: `claude-haiku-5-5`. Agent: verdict policy `strict`, tool budget 6.

| Setting | Completed | Accuracy (95% CI) | Macro-F1 (95% CI) | Mean tokens | Mean s |
| --- | ---: | ---: | ---: | ---: | ---: |
| closed_book | 50/50 | 48.0% (34.0% to 62.0%) | 45.9% (31.3% to 58.6%) | 576.02 | 2.923 |
| agent | 50/50 | 24.0% (12.0% to 36.0%) | 21.5% (11.1% to 32.6%) | 22595.3 | 31.805 |

Committed answers (yes or no, as opposed to maybe):

| Setting | Commits to yes/no | Accuracy when committed (95% CI) |
| --- | ---: | ---: |
| closed_book | 64.0% (32/50) | 59.4% (40.6% to 75.0%) |
| agent | 26.0% (13/50) | 38.5% (15.4% to 61.5%) |

Paired accuracy differences on the same questions:

- agent minus closed_book: -24.0 points (95% CI -38.0 to -10.0, 50 questions): a clear difference.

Agent behaviour:

- Mean tool calls: 5.9, mean cited claims: 3.08
- Answers where the verifier removed a claim: 22.0%
- Answers of insufficient_evidence (scored as maybe): 28.0%
- Questions where the source paper reached the evidence: 0 (must be 0)

Settings: closed_book answers from model knowledge only. with_context answers from the paper's own abstract without its conclusion (the standard PubMedQA setting). agent searches PubMed itself, with the source paper and everything from its year onwards excluded.
