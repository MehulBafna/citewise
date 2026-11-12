# PubMedQA evaluation

Generated 2026-11-12T12:47:22+00:00. Split: test, 100 questions (seed 0). Main model: `claude-sonnet-5-5`, fast model: `claude-haiku-5-5`. Agent: verdict policy `strict`, tool budget 6.

| Setting | Completed | Accuracy (95% CI) | Macro-F1 (95% CI) | Mean tokens | Mean s |
| --- | ---: | ---: | ---: | ---: | ---: |
| closed_book | 100/100 | 50.0% (40.0% to 60.0%) | 45.1% (35.8% to 54.2%) | 577.09 | 2.781 |
| with_context | 100/100 | 77.0% (69.0% to 85.0%) | 72.6% (62.4% to 81.4%) | 1027.85 | 2.438 |
| find_and_read | 100/100 | 49.0% (39.0% to 59.0%) | 48.0% (38.6% to 57.0%) | 17291.44 | 35.397 |

Committed answers (yes or no, as opposed to maybe):

| Setting | Commits to yes/no | Accuracy when committed (95% CI) |
| --- | ---: | ---: |
| closed_book | 71.0% (71/100) | 63.4% (52.1% to 74.6%) |
| with_context | 83.0% (83/100) | 81.9% (73.5% to 90.4%) |
| find_and_read | 57.0% (57/100) | 71.9% (59.6% to 82.5%) |

Paired accuracy differences on the same questions:

- find_and_read minus with_context: -28.0 points (95% CI -37.0 to -19.0, 100 questions): a clear difference.
- find_and_read minus closed_book: -1.0 points (95% CI -13.0 to +11.0, 100 questions): the data cannot tell them apart.
- with_context minus closed_book: +27.0 points (95% CI +16.0 to +38.0, 100 questions): a clear difference.

Agent behaviour (find_and_read):

- Mean tool calls: 4.45, mean cited claims: 3.44
- Answers where the verifier removed a claim: 26.0%
- Answers of insufficient_evidence (scored as maybe): 8.0%
- Questions where the agent found the source paper: 84.0%
- Questions where forbidden text reached the evidence: 0 (must be 0)

Settings: closed_book answers from model knowledge only. with_context answers from the paper's own abstract without its conclusion (the standard PubMedQA setting). find_and_read must find that paper in PubMed itself and sees only the same abstract without conclusion; other papers must be from earlier years. agent has the source paper excluded and only earlier papers allowed, so it must forecast the result from prior literature.
