# PubMedQA evaluation

Generated 2026-11-12T13:31:24+00:00. Split: dev, 50 questions (seed 0). Main model: `claude-sonnet-5-5`, fast model: `claude-haiku-5-5`. Agent: verdict policy `strict`, tool budget 6.

| Setting | Completed | Accuracy (95% CI) | Macro-F1 (95% CI) | Mean tokens | Mean s |
| --- | ---: | ---: | ---: | ---: | ---: |
| with_context | 50/50 | 74.0% (62.0% to 86.0%) | 66.4% (53.5% to 79.2%) | 1003.56 | 2.309 |
| find_and_read | 50/50 | 78.0% (66.0% to 88.0%) | 71.9% (56.9% to 84.6%) | 19984.0 | 43.794 |

Committed answers (yes or no, as opposed to maybe):

| Setting | Commits to yes/no | Accuracy when committed (95% CI) |
| --- | ---: | ---: |
| with_context | 82.0% (41/50) | 82.9% (70.7% to 92.7%) |
| find_and_read | 86.0% (43/50) | 81.4% (69.8% to 93.0%) |

Paired accuracy differences on the same questions:

- find_and_read minus with_context: +4.0 points (95% CI -4.0 to +12.0, 50 questions): the data cannot tell them apart.

Agent behaviour (find_and_read):

- Mean tool calls: 6.64, mean cited claims: 4.12
- Answers where the verifier removed a claim: 28.0%
- Answers of insufficient_evidence (scored as maybe): 0.0%
- Answers with a study verdict: 98.0%
- Questions where the agent found the source paper: 100.0%
- Questions where it named the source paper as the study asked about: 98.0%
- Questions where forbidden text reached the evidence: 0 (must be 0)

Settings: closed_book answers from model knowledge only. with_context answers from the paper's own abstract without its conclusion (the standard PubMedQA setting). find_and_read must find that paper in PubMed itself and sees only the same abstract without conclusion; other papers must be from earlier years. agent has the source paper excluded and only earlier papers allowed, so it must forecast the result from prior literature.
