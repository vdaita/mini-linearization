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

    for method in results:
        plt.plot(results[method], label=method)
        y_values = np.array(results[method])
        x_values = np.arange(1, len(y_values) + 1)

        idx_95 = np.where(y_values >= 0.95)[0]
        hit_95 = str(x_values[idx_95[0]]) if len(idx_95) > 0 else "N/A"

        idx_90 = np.where(y_values >= 0.9)[0]
        hit_90 = str(x_values[idx_90[0]]) if len(idx_90) > 0 else "N/A"

        table.add_row(method, hit_95, hit_90)

    print(table)

    plt.xlabel('Number of Blocks')
    plt.ylabel('Percentage of attention score covered')
    plt.title('Block Selection Evaluation')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    fire.Fire(main)