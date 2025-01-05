import json
import numpy as np
from rich import print
from rich.table import Table
import matplotlib.pyplot as plt
import fire

def main(filepath: str):
    results = json.load(
        open(filepath, "r")
    )

    # Draw horizontal dashed lines
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axhline(y=0.9, color='b', linestyle='--')

    table = Table(title="Method Hitting Percentage Levels")
    table.add_column("Method", justify="center")
    table.add_column("Hits 0.95 at x", justify="center")
    table.add_column("Hits 0.9 at x", justify="center")
    table.add_column("Relative to Average 0.95", justify="center")
    table.add_column("Relative to Average 0.9", justify="center")


    avg_idx_95 = np.where(np.array(results["averaged"]) >= 0.95)[0][0]
    avg_idx_90 = np.where(np.array(results["averaged"]) >= 0.9)[0][0]

    for method in results:
        plt.plot(results[method], label=method)
        y_values = np.array(results[method])
        x_values = np.arange(1, len(y_values) + 1)

        idx_95 = np.where(y_values >= 0.95)[0][0]
        pct_95 = (idx_95 + 1) / len(y_values) * 100
        hit_95 = f"{pct_95:.2f}%" if len(y_values) > 0 else "N/A"
        

        idx_90 = np.where(y_values >= 0.9)[0][0]
        pct_90 = (idx_90 + 1) / len(y_values) * 100
        hit_90 = f"{pct_90:.2f}%" if len(y_values) > 0 else "N/A"

        rel_avg_95 = (idx_95 - avg_idx_95) / avg_idx_95 * 100
        rel_avg_90 = (idx_90 - avg_idx_90) / avg_idx_90 * 100
        rel_avg_95 = f"[green]{rel_avg_95:.2f}%[green]" if rel_avg_95 <= 0 else f"[red]{rel_avg_95:.2f}%[red]"
        rel_avg_90 = f"[green]{rel_avg_90:.2f}%[green]" if rel_avg_90 <= 0 else f"[red]{rel_avg_90:.2f}%[red]"

        table.add_row(method, hit_95, hit_90, rel_avg_95, rel_avg_90)

    print(table)

    plt.xlabel('Number of Blocks')
    plt.ylabel('Percentage of attention score covered')
    plt.title('Block Selection Evaluation')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    fire.Fire(main)