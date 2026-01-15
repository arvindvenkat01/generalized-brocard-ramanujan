# Generalized Brocard–Ramanujan Problem: Search and Visualization

This repository contains the computational search algorithms and visualization tools for the paper:

**Perfect Squares from Sums of Consecutive Factorials: An Exceptional Solution in a Generalized Brocard-Ramanujan Family**

### Pre-print (Zenodo): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17137045.svg)](https://doi.org/10.5281/zenodo.17137045)
* **DOI** - 10.5281/zenodo.17137045
* **URL** - https://doi.org/10.5281/zenodo.17137045

## Abstract
The classical Brocard-Ramanujan problem, $n!+1=k^{2}$, is a long-standing open problem in number theory. This work explores the generalized Diophantine equation $\sum_{i=0}^{a}(n+i)!+1=k^{2}$.

The paper reports the discovery of a remarkable new solution, $(n,k,a)=(4,215,4)$, found through a systematic computational search. This repository provides the high-performance Python script used to verify solutions up to $n=30,000$ and $a=100$, as well as the tools to reproduce the theoretical magnitude curve (Figure 1).

## Features
- **High-Performance Search:** Efficiently scans for solutions using `gmpy2` for arbitrary-precision arithmetic and fast square detection.
- **Verification:** Exhaustively searches the range $1 \le n \le 30,000$ and $0 \le a \le 100$.
- **Visualization:** Generates the theoretical magnitude curve comparing the 7 known solutions against the approximation $k \approx \sqrt{(n+a)!}$.
- **Transparency:** Logs solutions and classifications (Brocard, Consecutive Pair, or Exceptional) in real-time.

## Repository Contents
- `search.py` : The main script. Performs the search and automatically generates the plot.
- `solutions.txt` : The structured dataset containing all found solutions.
- `console_output_search.txt` : Raw terminal logs from the $n=30,000$ run, verifying performance and results.
- `magnitude_curve.pdf` : The generated figure used in the paper.
- `README.md` : This documentation file.
- `requirements.txt` : Python dependencies.

## Requirements
- **Python 3.8+**
- **gmpy2** (Required for fast arbitrary-precision arithmetic)
- **numpy**, **matplotlib** (Required for plotting)

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/arvindvenkat01/generalized-brocard-ramanujan.git](https://github.com/arvindvenkat01/generalized-brocard-ramanujan.git)
   cd generalized-brocard-ramanujan
   ```
   
2. **Install the required dependencies:** (Note: gmpy2 requires GMP/MPFR/MPC libraries installed on your system)
   ```bash
   pip install gmpy2 numpy matplotlib
   ```

## Usage

### Run the Search
To reproduce the findings from the paper (this takes a few minutes depending on hardware):
```bash
python search.py
```

This will:
* Search for solutions up to $n=30,000$ and $a=100$.
* Print progress and discovered solutions to the console.
* Save the results to solutions.txt.
* Generate the visualization as magnitude_curve.pdf.

### Custom Search
You can override the search bounds via command-line arguments:
```bash
python search.py --max_n 50000 --max_a 20
```

## Citation



If you use this work, please cite the paper using the Zenodo archive.


---



## License



The content of this repository is dual-licensed:

- **MIT License** for `search.py` See the [LICENSE](LICENSE) file for details.
- **CC BY 4.0** (Creative Commons Attribution 4.0 International) for all other content (results.txt, README, etc.)

