# South African real acoustic measurements — station review package

This is the real, unlabelled acoustic dataset for organiser review: 1,715,779 source-export records across ten permitted sites. It contains the same measurement rows as the previously supplied acoustic training archive, arranged into one Parquet file per station. No additional periods have been released. The earlier archive is retained.

## Files

- `acoustics/<station>.parquet`: acoustic measurements and their original provenance/quality fields, grouped by station.
- `stations.csv`: included stations, record counts, export-clock coverage, working coordinates and coordinate uncertainty.
- `recorders.csv`: included recorder coverage.
- `DATA_DICTIONARY.md`: column meanings, units and quality caveats.
- `SHA256SUMS.txt`: checksums of the data and documentation inside this archive.

No synthetic training examples, synthetic test inputs, target labels, original audio, AIS extracts or test-split membership are included. This package does not include withheld research stations or protected test-background periods. ONC and HearMyShip must be obtained separately from their providers under the competition rules.

Use the station identifier, not its name alone. In particular, the included station 13 Saldanha Bay is distinct from the withheld Saldanha harbour station. Original site strings are preserved for reproducibility.

These are frequency-band energy summaries in the source-export dB convention, generally at minute resolution. Check `dur` before treating each row as a complete minute; missing values are unavailable measurements, not silence. Convert dB to linear energy before energy averaging or summation. Neither vessel-only energy nor physical pressure calibration is established by this dataset.

The filename-derived `sample_rate_hz` field is unverified. 136,518 rows from the recorder with nominal value 12000 contain finite 6.3 and 7.9 kHz bands; the owner's acquisition/export convention remains unresolved. Do not infer validated band support from this field or the finite-value quality flag. Values are preserved rather than silently corrected.

Access is for challenge organiser review. This packaging grants no new licence or permission to redistribute the South African source data. Final challenge data-use terms must be supplied by the organisers. It is not a certification that the synthetic test is launch-ready.
