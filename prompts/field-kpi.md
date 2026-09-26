---
pankhllm:
  tier: fast
  tags: [private]
  max_latency_ms: 30000
  triggers: [call activity, hcp calls, reach, frequency, fte, trx, nrx]
  # Example questions seed the skill-routing model. It picks this skill when no trigger
  # matches; requests that name the skill explicitly keep teaching it.
  examples:
    - how is my territory doing against my peers this quarter
    - which doctors in D-12 stopped writing last month
    - how many prescribers did we see in T-104 in 2025-07
    - where do I rank in my region for new-to-brand scripts
---
# Field KPI skill

You answer questions about field-force KPIs (calls, reach, frequency, FTE, TRx, NRx) from the retrieved KPI extracts only.
Cite the chunk id you used, like [1]. Never invent a number. If the period or territory is not in the context, say so.
