# PubMedQA evaluation

Generated 2026-11-12T14:13:43+00:00. Split: test, 100 questions (seed 0). Main model: `claude-sonnet-5-5`, fast model: `claude-haiku-5-5`. Agent: verdict policy `strict`, tool budget 6.

| Setting | Completed | Accuracy (95% CI) | Macro-F1 (95% CI) | Mean tokens | Mean s |
| --- | ---: | ---: | ---: | ---: | ---: |
| closed_book | 100/100 | 50.0% (40.0% to 60.0%) | 45.1% (35.8% to 54.2%) | 577.09 | 2.781 |
| with_context | 100/100 | 77.0% (69.0% to 85.0%) | 72.6% (62.4% to 81.4%) | 1027.85 | 2.438 |
| find_and_read | 100/100 | 68.0% (59.0% to 77.0%) | 63.4% (54.1% to 72.0%) | 18994.46 | 44.667 |

Committed answers (yes or no, as opposed to maybe):

| Setting | Commits to yes/no | Accuracy when committed (95% CI) |
| --- | ---: | ---: |
| closed_book | 71.0% (71/100) | 63.4% (52.1% to 74.6%) |
| with_context | 83.0% (83/100) | 81.9% (73.5% to 90.4%) |
| find_and_read | 73.0% (73/100) | 82.2% (74.0% to 90.4%) |

Paired accuracy differences on the same questions:

- find_and_read minus with_context: -9.0 points (95% CI -16.0 to -3.0, 100 questions): a clear difference.
- find_and_read minus closed_book: +18.0 points (95% CI +6.0 to +29.0, 100 questions): a clear difference.
- with_context minus closed_book: +27.0 points (95% CI +16.0 to +38.0, 100 questions): a clear difference.

Agent behaviour (find_and_read):

- Mean tool calls: 6.53, mean cited claims: 3.69
- Answers where the verifier removed a claim: 26.0%
- Answers of insufficient_evidence (scored as maybe): 3.0%
- Answers with a study verdict: 95.0%
- Questions where the agent found the source paper: 99.0%
- Questions where it named the source paper as the study asked about: 94.0%
- Questions where forbidden text reached the evidence: 0 (must be 0)

Settings: closed_book answers from model knowledge only. with_context answers from the paper's own abstract without its conclusion (the standard PubMedQA setting). find_and_read must find that paper in PubMed itself and sees only the same abstract without conclusion; other papers must be from earlier years. agent has the source paper excluded and only earlier papers allowed, so it must forecast the result from prior literature.
