import json
import matplotlib.pyplot as plt

results = json.load(
    open("bsa_results_64.json", "r")
)

for method in results:
    plt.plot(results[method], label=method)

plt.xlabel('Number of Blocks')
plt.ylabel('Percentage of attention score covered')
plt.title('Block Selection Evaluation')
plt.legend()
plt.show()