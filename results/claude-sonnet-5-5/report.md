# PubMedQA evaluation

Generated 2026-11-12T11:10:08+00:00. Split: test, 30 questions (seed 0). Main model: `claude-sonnet-5-5`, fast model: `claude-haiku-5-5`.

| Setting | Completed | Accuracy (95% CI) | Macro-F1 (95% CI) | Mean tokens | Mean s |
| --- | ---: | ---: | ---: | ---: | ---: |
| closed_book | 30/30 | 63.3% (46.7% to 80.0%) | 58.8% (41.6% to 74.5%) | 583.133 | 3.029 |
| with_context | 30/30 | 86.7% (73.3% to 96.7%) | 82.4% (60.6% to 96.4%) | 1022.767 | 2.722 |
| agent | 30/30 | 50.0% (33.3% to 66.7%) | 50.3% (32.8% to 66.1%) | 24068.767 | 31.591 |

Paired accuracy differences on the same questions:

- agent minus closed_book: -13.3 points (95% CI -33.3 to +6.7, 30 questions): the data cannot tell them apart.
- with_context minus closed_book: +23.3 points (95% CI +3.3 to +43.3, 30 questions): a clear difference.
- agent minus with_context: -36.7 points (95% CI -56.7 to -16.7, 30 questions): a clear difference.

Agent behaviour:

- Mean tool calls: 6.0, mean cited claims: 2.967
- Answers where the verifier removed a claim: 16.7%
- Answers of insufficient_evidence (scored as maybe): 36.7%
- Questions where the source paper reached the evidence: 0 (must be 0)

Settings: closed_book answers from model knowledge only. with_context answers from the paper's own abstract without its conclusion (the standard PubMedQA setting). agent searches PubMed itself, with the source paper and everything from its year onwards excluded.
