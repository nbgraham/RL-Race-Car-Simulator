from matplotlib import pyplot as plt

def main():
    a = [100-99*i/500 for i in range(500)]
    b = [1-0.9*(i-500)/400 for i in range (500,900)]
    c = [0.1]*100

    a.extend(b)
    a.extend(c)

    plt.plot(a)
    plt.yscale('log')
    plt.title('Temperature over episodes (logarithmic in y)')
    plt.ylabel('Temperature')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == "__main__":
    main()