# Experiment ladder results

## MQAR: SigmaStorm vs a size-matched transformer

Task: MQAR 26 bindings / 16 values / 4 queries; chance 0.062.

| Model | Params | Accuracy | Train s |
| :--- | ---: | ---: | ---: |
| sigma | 2,019,039 | 0.977 | 137.1 |
| transformer | 622,548 | 0.096 | 25.2 |

## Recall vs distance to the binding

Task: delayed recall, 8 bindings, length 320.

| Variant | Accuracy | <32 | 32-127 | 128-255 | >=256 | Params |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.984 | 0.984 | 0.984 | 1.000 | 0.969 | 2,084,840 |
| delta_spectrum | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1,513,132 |
| delta_single_rate | 0.758 | 0.984 | 1.000 | 0.750 | 0.297 | 1,513,132 |

## Global aggregate path (chunk summaries)

Task: count symbols seen ~150 tokens back.

| Variant | Accuracy | Params |
| :--- | ---: | ---: |
| full | 0.703 | 2,080,744 |
| no_summary | 0.672 | 1,817,556 |

## State capacity: state_expand

Task: dense MQAR, 66 bindings, 4 queries.

| Variant | Accuracy | Params | State bytes |
| :--- | ---: | ---: | ---: |
| expand_1 | 0.754 | 1,694,911 | 167,936 |
| expand_2 | 0.555 | 2,027,231 | 200,704 |

## Slot write granularity

Task: presence recall, length 192, conv + slots only.

| Variant | Accuracy | Params | State bytes |
| :--- | ---: | ---: | ---: |
| slot_block_64 | 0.438 | 1,155,984 | 85,376 |
| slot_block_4096 | 0.453 | 1,155,984 | 85,376 |

## Cost at equal width

| Config | Sigma params | Sigma forward s | Transformer params | Transformer forward s |
| :--- | ---: | ---: | ---: | ---: |
| tiny_T128 | 2,076,383 | 0.031 | 679,892 | 0.0036 |
| tiny_T512 | 2,076,383 | 0.1021 | 679,892 | 0.0147 |
| tiny_T2048 | 2,076,383 | 0.4111 | 679,892 | 0.1404 |
| 20m_T128 | 20,965,370 | 0.0938 | 8,369,604 | 0.018 |
| 20m_T512 | 20,965,370 | 0.3214 | 8,369,604 | 0.0761 |
| 20m_T2048 | 20,965,370 | 1.3388 | 8,369,604 | 0.5872 |
