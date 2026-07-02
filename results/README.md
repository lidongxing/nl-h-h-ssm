# results/

Table 6 training logs: **50 JSON files**, each with a **unique** role and final metrics.

See also [docs/descriptions/results/](../docs/descriptions/results/) for one stub per file.

| File | Description |
|------|-------------|
| `results/electricity_informer.json` | Electricity-L Informer — RMSSE **0.74**, **best** Elec. column. Metrics: smape=30.1664, rmsse=0.7442, acd=22.7080. |
| `results/electricity_mamba1.json` | Electricity-L Mamba-1 — RMSSE 1.70, third place. Metrics: smape=56.3981, rmsse=1.6950, acd=21.0907. |
| `results/electricity_mamba2.json` | Electricity-L Mamba-2 — RMSSE 1.42, underline second. Metrics: smape=52.3709, rmsse=1.4185, acd=21.8865. |
| `results/electricity_nlh_ssm.json` | Electricity-L NL-H-H-SSM — RMSSE 2.33; flat ACD 0.368. Metrics: smape=199.8618, rmsse=2.3278, acd=0.3679. |
| `results/electricity_transformer.json` | Electricity-L Transformer — RMSSE 2.39, mid industrial baseline. Metrics: smape=198.9267, rmsse=2.3891, acd=0.3679. |
| `results/labor_informer.json` | Labour-AU Informer — sMAPE 176.35, weakest sMAPE among five models. Metrics: smape=176.3545, rmsse=14.9155, acd=0.5615. |
| `results/labor_mamba1.json` | Labour-AU Mamba-1 — sMAPE 139.83, underline second on Lab. column. Metrics: smape=139.8272, rmsse=17.5164, acd=0.3386. |
| `results/labor_mamba2.json` | Labour-AU Mamba-2 — sMAPE 53.60, **best** Lab. sMAPE by large margin. Metrics: smape=53.6036, rmsse=5.1114, acd=13.3978. |
| `results/labor_nlh_ssm.json` | Labour-AU NL-H-H-SSM — sMAPE 195.03; prioritizes ACD 0.368 over sMAPE here. Metrics: smape=195.0333, rmsse=15.5070, acd=0.3679. |
| `results/labor_transformer.json` | Labour-AU Transformer — sMAPE 141.10, competitive but not best. Metrics: smape=141.0965, rmsse=17.0130, acd=0.3679. |
| `results/logic_informer.json` | LogicGraph Informer — ACD 0.511, weaker structure fidelity. Metrics: smape=176.3466, rmsse=0.9785, acd=0.5110. |
| `results/logic_mamba1.json` | LogicGraph Mamba-1 — ACD **0.326**, **best** Logic. structure column. Metrics: smape=113.5451, rmsse=1.2823, acd=0.3264. |
| `results/logic_mamba2.json` | LogicGraph Mamba-2 — ACD 8.23, collapsed hierarchy embedding. Metrics: smape=125.2674, rmsse=0.9285, acd=8.2271. |
| `results/logic_nlh_ssm.json` | LogicGraph NL-H-H-SSM — ACD 0.368 tied best; saturates ablation sweeps. Metrics: smape=199.5630, rmsse=0.9535, acd=0.3679. |
| `results/logic_transformer.json` | LogicGraph Transformer — ACD 0.368 tied best; sMAPE 175.2. Metrics: smape=175.2077, rmsse=0.9556, acd=0.3679. |
| `results/m5_informer.json` | M5-Walmart Informer — RMSSE 7.05, weakest RMSSE on M5. Metrics: smape=116.1499, rmsse=7.0540, acd=6.7544. |
| `results/m5_mamba1.json` | M5-Walmart Mamba-1 — RMSSE 87.56, unstable scale on this hierarchy. Metrics: smape=165.3939, rmsse=87.5648, acd=0.7416. |
| `results/m5_mamba2.json` | M5-Walmart Mamba-2 — RMSSE 28.93, mid-tier among baselines. Metrics: smape=167.2794, rmsse=28.9346, acd=0.8203. |
| `results/m5_nlh_ssm.json` | M5-Walmart NL-H-H-SSM — RMSSE **4.30**, **best** M5 column in Table 6. Metrics: smape=199.6964, rmsse=4.2985, acd=0.3679. |
| `results/m5_transformer.json` | M5-Walmart Transformer — RMSSE 5.40, underline **second**; strong industrial baseline. Metrics: smape=199.6933, rmsse=5.3983, acd=0.3679. |
| `results/medical_informer.json` | Med-Diag-Path Informer — ACD 0.470, third on structure. Metrics: smape=145.3700, rmsse=1.6959, acd=0.4700. |
| `results/medical_mamba1.json` | Med-Diag-Path Mamba-1 — ACD **0.366**, **best** Med. ACD (underline). Metrics: smape=185.8735, rmsse=1.7448, acd=0.3664. |
| `results/medical_mamba2.json` | Med-Diag-Path Mamba-2 — ACD 9.58, poor cophenetic fit. Metrics: smape=55.7301, rmsse=1.0081, acd=9.5771. |
| `results/medical_nlh_ssm.json` | Med-Diag-Path NL-H-H-SSM — ACD 0.368 tied best; RMSSE 48.3 outlier. Metrics: smape=195.6526, rmsse=48.3269, acd=0.3679. |
| `results/medical_transformer.json` | Med-Diag-Path Transformer — ACD 0.368 tied best; sMAPE 131.0. Metrics: smape=130.9711, rmsse=2.1845, acd=0.3679. |
| `results/prison_informer.json` | Prison-AU Informer — sMAPE 155.80, underline second on Pris. Metrics: smape=155.8031, rmsse=0.9015, acd=0.1970. |
| `results/prison_mamba1.json` | Prison-AU Mamba-1 — sMAPE 172.61, worst sMAPE among five. Metrics: smape=172.6116, rmsse=0.9230, acd=0.3499. |
| `results/prison_mamba2.json` | Prison-AU Mamba-2 — sMAPE 113.23, **best** Pris. sMAPE. Metrics: smape=113.2322, rmsse=0.7579, acd=1.7049. |
| `results/prison_nlh_ssm.json` | Prison-AU NL-H-H-SSM — sMAPE 193.87; ACD 0.368 tied best expert metric. Metrics: smape=193.8717, rmsse=0.8657, acd=0.3679. |
| `results/prison_transformer.json` | Prison-AU Transformer — sMAPE 165.17, mid-pack on Pris. Metrics: smape=165.1663, rmsse=0.9516, acd=0.3679. |
| `results/solar_informer.json` | Solar-HTS Informer — RMSSE **0.46**, **best** Solr. column. Metrics: smape=6.9950, rmsse=0.4565, acd=10.0748. |
| `results/solar_mamba1.json` | Solar-HTS Mamba-1 — RMSSE 0.89, third. Metrics: smape=14.3896, rmsse=0.8941, acd=10.1385. |
| `results/solar_mamba2.json` | Solar-HTS Mamba-2 — RMSSE 0.76, underline second. Metrics: smape=12.1347, rmsse=0.7635, acd=10.2530. |
| `results/solar_nlh_ssm.json` | Solar-HTS NL-H-H-SSM — RMSSE 8.92; high sMAPE 199.5 on this stem. Metrics: smape=199.5264, rmsse=8.9229, acd=0.3679. |
| `results/solar_transformer.json` | Solar-HTS Transformer — RMSSE 8.92, tied NL-H worst tier. Metrics: smape=199.2538, rmsse=8.9229, acd=0.3679. |
| `results/tourism_informer.json` | Tourism-AU Informer — sMAPE 163.42, second-best socio-economic forecaster here. Metrics: smape=163.4238, rmsse=0.7435, acd=1.7612. |
| `results/tourism_mamba1.json` | Tourism-AU Mamba-1 — sMAPE 156.44, underline (2nd) in Table 6 Tour. column. Metrics: smape=156.4442, rmsse=0.7320, acd=0.3675. |
| `results/tourism_mamba2.json` | Tourism-AU Mamba-2 — sMAPE 151.09, **best** Tour. sMAPE in committed runs. Metrics: smape=151.0895, rmsse=0.7279, acd=1.6905. |
| `results/tourism_nlh_ssm.json` | Tourism-AU NL-H-H-SSM — sMAPE 196.36; structure metric ACD 0.368 tied best. Metrics: smape=196.3622, rmsse=0.7359, acd=0.3679. |
| `results/tourism_transformer.json` | Tourism-AU Transformer baseline — sMAPE 171.47 (rank 4/5 on Tour.). Metrics: smape=171.4690, rmsse=0.7448, acd=0.3679. |
| `results/traffic_informer.json` | Traffic-HTS Informer — RMSSE **0.47**, **best** Traf. column. Metrics: smape=27.6481, rmsse=0.4748, acd=19.3969. |
| `results/traffic_mamba1.json` | Traffic-HTS Mamba-1 — RMSSE 0.94, third. Metrics: smape=56.8156, rmsse=0.9395, acd=20.9127. |
| `results/traffic_mamba2.json` | Traffic-HTS Mamba-2 — RMSSE 0.70, underline second. Metrics: smape=43.8564, rmsse=0.7036, acd=20.5305. |
| `results/traffic_nlh_ssm.json` | Traffic-HTS NL-H-H-SSM — RMSSE 1.97; ACD 0.368 tied best. Metrics: smape=198.6918, rmsse=1.9666, acd=0.3679. |
| `results/traffic_transformer.json` | Traffic-HTS Transformer — RMSSE 1.97, tied with NL-H on Traf. Metrics: smape=197.9427, rmsse=1.9675, acd=0.3679. |
| `results/wiki_informer.json` | Wiki-Traffic Informer — RMSSE **0.60**, **best** Wiki column. Metrics: smape=66.4237, rmsse=0.6043, acd=10.1258. |
| `results/wiki_mamba1.json` | Wiki-Traffic Mamba-1 — RMSSE 0.90, third among five. Metrics: smape=107.0807, rmsse=0.9033, acd=6.2958. |
| `results/wiki_mamba2.json` | Wiki-Traffic Mamba-2 — RMSSE 0.85, underline second. Metrics: smape=109.9622, rmsse=0.8514, acd=5.4257. |
| `results/wiki_nlh_ssm.json` | Wiki-Traffic NL-H-H-SSM — RMSSE 1.12; ACD 0.368 tied best structure score. Metrics: smape=199.0289, rmsse=1.1178, acd=0.3679. |
| `results/wiki_transformer.json` | Wiki-Traffic Transformer — RMSSE 1.12, tied NL-H on Wiki. Metrics: smape=199.2160, rmsse=1.1178, acd=0.3679. |

Regenerate this table: `python scripts/generate_repo_manifest.py`
