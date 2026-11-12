# PubMedQA evaluation

Generated 2026-11-12T11:58:55+00:00. Split: dev, 50 questions (seed 0). Main model: `claude-sonnet-5-5`, fast model: `claude-haiku-5-5`. Agent: verdict policy `best_supported`, tool budget 6.

| Setting | Completed | Accuracy (95% CI) | Macro-F1 (95% CI) | Mean tokens | Mean s |
| --- | ---: | ---: | ---: | ---: | ---: |
| agent | 50/50 | 28.0% (16.0% to 40.0%) | 26.0% (15.4% to 35.9%) | 26153.48 | 33.65 |

Committed answers (yes or no, as opposed to maybe):

| Setting | Commits to yes/no | Accuracy when committed (95% CI) |
| --- | ---: | ---: |
| agent | 40.0% (20/50) | 45.0% (25.0% to 65.0%) |

Agent behaviour:

- Mean tool calls: 5.94, mean cited claims: 4.3
- Answers where the verifier removed a claim: 34.0%
- Answers of insufficient_evidence (scored as maybe): 18.0%
- Questions where the source paper reached the evidence: 0 (must be 0)

Settings: closed_book answers from model knowledge only. with_context answers from the paper's own abstract without its conclusion (the standard PubMedQA setting). agent searches PubMed itself, with the source paper and everything from its year onwards excluded.
