# CGBench

CGBench is a benchmarking framework for evaluation of scientific reasoning in language models (LMs). 
CGBench leverages ClinGen (clinicalgenome.org), an extensive repository of clinical genetics annotations and interpretations of literature for gene-disease associations and variant annotations, to evaluate the ability of LMs to extract, interpret, and explain fine-grained results from scientific publications.
We formulate three separate tasks in CGBench, and our framework leverages both classification-based metrics as well as LM-as-a-judge approaches to holistically evaluate LMs.
Stay tuned for the release of our paper!

## 🚀 Quick‑start: reproduce the exact environment

To quick-start, you'll use [uv](https://docs.astral.sh/uv/), a fast and efficient Python package manager. This installs the necessary environment

```bash
# Install uv (faster alternative to pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/owencqueen/cgbench.git
cd cgbench

# Clone and create env:
uv venv --python=3.12          # or: python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Sync to the exact locked versions:
uv sync       # reads uv.lock, installs with hashes
```

## ♻️ Reproducibility

All scripts for each task are found in the following directories::

- VCI Evidence Scoring: `clingen_vci/evidence_scoring/`
- VCI Evidence Verification: `clingen_vci/evidence_sufficiency/`
- GCI Evidence Extraction: `clingen_gci/`

Stay tuned for each scripts to reproduce each experiment in the paper.
