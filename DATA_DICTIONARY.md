# Data dictionary

| Field | Meaning |
|---|---|
| `site` | Source site identifier and name. Use the leading numeric ID to distinguish sites. |
| `recorder`, `station_token` | Export-derived recorder/provenance identifiers. They do not certify deployment coordinates. |
| `src_file`, `file_kind`, `file_date` | Original export filename, export type and filename date. |
| `sample_rate_hz` | Numeric token inferred from the export filename; unverified acquisition/export setting, not a measured WAV header rate. |
| `t` | Original acoustic export timestamp. Timezone/UTC alignment has not been verified; no timezone conversion was applied. |
| `dur` | Observed duration in seconds. Range 1–360; 36,234 rows differ from 60 seconds and 16,896 are shorter than 55 seconds. |
| `TOL_<frequency>` | Band level in the source-export dB convention; frequency suffix in Hz. Original columns, values and missing observations are preserved. Availability and calibration differ between recorders. |
| `qc_duration_ge55` | Duration screening indicator, not a complete validity decision. |
| `qc_25band_finite` | All 25 common bands have finite values; does not establish frequency support, calibration or vessel presence. |
| `qc_25band_spread_ge6` | Common-band range is at least 6 dB; a screening indicator, not a vessel label. |

The common 25 bands span nominal centres 31.6–7943.3 Hz. Other export columns are retained for research. Original acoustic waveforms and verified vessel labels are unavailable. Check gaps, duration, sentinel values and recorder changes before constructing features or plots.

Station coordinates are working site-level candidates. Station 13 has two alternative coordinates with unresolved deployment assignment. `stations.csv` preserves these candidates and notes; they must not be treated as verified recorder positions. Acoustic timestamps retain the export clock. If external AIS is used, independently establish clock alignment and remember that AIS proximity is not an acoustic label.
