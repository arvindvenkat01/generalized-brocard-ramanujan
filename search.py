#!/usr/bin/env python3
"""
Generalized Brocard-Ramanujan Problem: Complete Search and Visualization
========================================================================

Systematic computational search for integer solutions to:
    Sum_{i=0}^{a} (n+i)! + 1 = k^2

This script implements the exhaustive search described in Section 5 of:
"Perfect Squares from Sums of Consecutive Factorials: An Exceptional Solution 
in a Generalized Brocard-Ramanujan Family" by Arvind N. Venkat.

Usage:
    python search.py                          # Runs default paper verification (n=30k, a=100)
    python search.py --max_n 50000            # Override bounds
    python search.py --output results/        # Custom output folder

Dependencies:
    - gmpy2 (Required for arbitrary-precision arithmetic)
    - numpy, matplotlib (Optional, for generating figures)

Copyright (c) 2026 Arvind N. Venkat
License: MIT
Repository: https://github.com/arvindvenkat01/generalized-brocard-ramanujan
"""

import sys
import time
import math
import argparse
from typing import List, Tuple, Optional
from pathlib import Path

# Essential Math Import
try:
    from gmpy2 import mpz, isqrt, is_square
except ImportError:
    print("CRITICAL ERROR: 'gmpy2' is missing.")
    print("Please install it using: pip install gmpy2")
    sys.exit(1)

# Optional Plotting Imports
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from math import lgamma
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: 'numpy' or 'matplotlib' missing. Figures will not be generated.")


def search_generalized_brocard(
    max_n: int,
    max_a: int,
    progress_interval: int = 500
) -> List[Tuple[int, int, int]]:
    """
    Exhaustively search for solutions to Σ(i=0 to a) (n+i)! + 1 = k².
    
    Args:
        max_n: Maximum n value (inclusive)
        max_a: Maximum a value (inclusive)  
        progress_interval: Progress reporting frequency
        
    Returns:
        List of (n, k, a) solutions
    """
    solutions = []
    start_time = time.time()
    
    print(f"╔{'═'*68}╗")
    print(f"║ Generalized Brocard-Ramanujan Search{' '*37}║")
    print(f"║ Bounds: n ≤ {max_n:,}, a ≤ {max_a:,}{' '*(44 - len(f'{max_n:,}') - len(f'{max_a:,}'))}║")
    print(f"╚{'═'*68}╝\n")
    
    factorial_n = mpz(1)
    
    for n in range(1, max_n + 1):
        factorial_n *= n
        
        # --- Case a=0 (Classical Brocard) ---
        if is_square(factorial_n + 1):
            k = int(isqrt(factorial_n + 1))
            solutions.append((n, k, 0))
            _print_solution(n, k, 0, time.time() - start_time, len(solutions))
        
        # --- Case a > 0 (Generalized) ---
        term_factorial = mpz(factorial_n)
        current_sum = mpz(factorial_n)
        
        for a in range(1, max_a + 1):
            # Calculate (n+a)! incrementally: (n+a)! = (n+a-1)! * (n+a)
            term_factorial *= (n + a)
            current_sum += term_factorial
            
            # Check Sum + 1
            if is_square(current_sum + 1):
                k = int(isqrt(current_sum + 1))
                solutions.append((n, k, a))
                _print_solution(n, k, a, time.time() - start_time, len(solutions))
        
        # Progress Reporting: Check at n=1 for immediate feedback, then intervals
        if n == 1 or n % progress_interval == 0:
            _print_progress(n, factorial_n, time.time() - start_time, max_n)
    
    _print_summary(solutions, time.time() - start_time)
    return solutions


def _print_solution(n: int, k: int, a: int, elapsed: float, count: int):
    """Print found solution with classification."""
    classifications = {
        0: "Brocard-Ramanujan (a=0)",
        1: "Consecutive Pair (a=1)", 
        4: "Exceptional Solution (a=4)"
    }
    classification = classifications.get(a, f"General (a={a})")
    
    print(f"\n{'!'*70}")
    print(f"  SOLUTION #{count}: (n={n}, k={k}, a={a})")
    print(f"  Classification: {classification}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'!'*70}\n")


def _print_progress(n: int, fact_n: mpz, elapsed: float, max_n: int):
    """Print performance metrics to stdout."""
    rate = n / elapsed if elapsed > 0 else 0
    # Optimization: Use num_digits() instead of len(str()) for speed
    digits = fact_n.num_digits() 
    print(f"n={n:>6,} ({100*n/max_n:>5.1f}%) | n! digits: {digits:>6,} | "
          f"Rate: {rate:>6.1f} n/s")


def _print_summary(solutions: List[Tuple[int, int, int]], elapsed: float):
    """Print final results summary."""
    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"Solutions Found: {len(solutions)}")
    print(f"Total Time:      {elapsed:.2f}s")
    print(f"{'='*70}")


def save_results(solutions: List[Tuple[int, int, int]], params: dict, output_dir: Path):
    """Save solutions to text file with metadata."""
    output_file = output_dir / "solutions.txt"
    with open(output_file, 'w') as f:
        f.write(f"# Generalized Brocard-Ramanujan Solutions\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Search Bounds: n <= {params['max_n']}, a <= {params['max_a']}\n")
        f.write(f"# Solutions Found: {len(solutions)}\n")
        f.write("# Format: n, k, a\n\n")
        for s in sorted(solutions):
            f.write(f"{s[0]}, {s[1]}, {s[2]}\n")
    print(f"✓ Results saved to: {output_file}")


def create_publication_figure(
    solutions: List[Tuple[int, int, int]],
    output_dir: Path
) -> Optional[Path]:
    """
    Generate Figure 1: Solutions on theoretical magnitude curve.
    """
    if not (PLOTTING_AVAILABLE and solutions):
        return None
    
    print("Generating publication figure...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # --- Theoretical curves ---
    x_vals = np.linspace(0.9, 9.1, 1000)
    
    # Exact Step Function
    y_exact = [np.log10(np.sqrt(math.factorial(int(val)) + 1)) for val in x_vals]
    
    # Smooth Approximation
    y_smooth = [0.5 * lgamma(val + 1) / np.log(10) for val in x_vals]
    
    ax.plot(x_vals, y_exact, 'k--', linewidth=1, dashes=(4, 2),
            label=r'Exact: $k = \sqrt{\lfloor n+a \rfloor! + 1}$', zorder=1)
    ax.plot(x_vals, y_smooth, color='gray', linestyle=(0, (1, 1.5)), linewidth=1.5,
            label=r'Approximation: $k \approx \sqrt{(n+a)!}$', zorder=1)
    
    # --- Solution points ---
    styles = {
        0: {'c': '#3498db', 'm': 'o', 'l': 'a=0 (Brocard-Ramanujan)'},
        1: {'c': '#2ecc71', 'm': 's', 'l': 'a=1 (Consecutive Pairs)'},
        4: {'c': '#e74c3c', 'm': '*', 'l': 'a=4 (Exceptional)', 's': 200}
    }
    
    plotted = set()
    for n, k, a in solutions:
        st = styles.get(a, {'c': '#9b59b6', 'm': 'D', 'l': f'a={a}'})
        label = st['l'] if st['l'] not in plotted else None
        plotted.add(st['l'])
        
        ax.scatter(n + a, np.log10(k), c=st['c'], marker=st['m'], 
                   s=st.get('s', 90), edgecolors='black', linewidth=1,
                   zorder=5, label=label)
        
        ax.annotate(f'k={k}', (n + a, np.log10(k)), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    
    # --- Formatting ---
    ax.set_xlabel(r'$x = n + a$ (Index of largest factorial term)', fontsize=12)
    ax.set_ylabel(r'$\log_{10}(k)$', fontsize=12)
    ax.set_title('All Known Solutions Follow the Theoretical Magnitude Curve',
                 fontsize=14, fontweight='bold')
                 
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0.5, 9.5)
    
    # Save
    pdf_path = output_dir / 'magnitude_curve.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    print(f"✓ Figure saved to: {pdf_path}")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Search for Generalized Brocard-Ramanujan solutions")
    parser.add_argument("--max_n", type=int, default=30000, help="Maximum n (inclusive)")
    parser.add_argument("--max_a", type=int, default=100, help="Maximum a (inclusive)")
    parser.add_argument("--output", type=Path, default=Path("."), help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    
    # Robust handling for Jupyter/Colab kernels which inject extra arguments (like -f)
    if 'ipykernel' in sys.modules:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"Note: ignoring injected notebook args: {unknown}")
    else:
        args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(exist_ok=True, parents=True)
    
    # --- EXECUTE SEARCH ---
    solutions = search_generalized_brocard(args.max_n, args.max_a)
    
    # --- SAVE RESULTS ---
    save_results(solutions, {'max_n': args.max_n, 'max_a': args.max_a}, args.output)
    
    # --- GENERATE PLOT ---
    if not args.no_plot:
        create_publication_figure(solutions, args.output)


if __name__ == "__main__":
    main()