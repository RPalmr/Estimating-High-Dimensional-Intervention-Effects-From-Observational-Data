Riboflavin data provenance
=========================

The paper uses 71 observations, 4,088 log gene expression covariates, and a log riboflavin production response from Bacillus subtilis. DSM supplied the original data.

The local `riboflavin.csv` is an unmodified CSV export retrieved through the GitHub connector from:

https://github.com/Zhelena/Analysis-of-Riboflavin-Production-in-Bacillus-subtilis/blob/b3f2913180cb7df779ef10c4204486df64db6987/riboflavin.csv

Git blob: `5c1e392f291d2190be06cea3d953d840db61a24a`

SHA-256: `41e7f85919fb2d355ae3a8c8bb0ea428ad74be06fe2f308e5c22324528f3cf0c`

The header has `y` first and 4,088 gene columns. There are 71 finite data rows. The runner places the response last, standardizes covariates with sample standard deviations, and keeps the response on its original scale. Each bootstrap resample is standardized separately.

The original authors' public R package provides the canonical distribution:

https://github.com/cran/hdi/blob/master/data/riboflavin.RData

The CSV mirror was necessary because this session could retrieve text through the connector but could not directly download binary R data. Its dimensions agree with the paper; an entrywise comparison against the canonical R data has not been performed. The source CSV is retained locally and excluded from Git to avoid assuming redistribution permission for a third-party export. A checksum-verified downloader is provided in `experiments/download_data.py`.
