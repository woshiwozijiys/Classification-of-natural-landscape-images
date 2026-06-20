import matplotlib.pyplot as plt

models = ["ViT1", "ViT2","ViT3"]
accuracies = [89.70, 92.73,94.60]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

for bar in bars:
    y = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        y + 1,
        f"{y:.2f}%",
        ha="center"
    )

plt.title("Comparison of Test Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig("compare.png")
plt.show()