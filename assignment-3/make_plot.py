import matplotlib.pyplot as plt


def main():
    x = [-200 for _ in range(2000)]
    plt.plot(x)
    plt.title("Score over episodes:")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.savefig("./a2c_plot")
    plt.close()

if __name__ == "__main__":
    main()
