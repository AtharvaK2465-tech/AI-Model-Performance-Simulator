import matplotlib.pyplot as plt

def plot_results(levels, rf_acc, lr_acc):
    plt.plot(levels, rf_acc, marker='o', label='Random Forest')
    plt.plot(levels, lr_acc, marker='s', label='Logistic Regression')

    plt.xlabel("Distortion Level")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid()

    plt.show()