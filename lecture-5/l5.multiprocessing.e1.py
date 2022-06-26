import multiprocessing as mp
import numpy as np


# Takes an array and squares each element
def square(x):
    return np.square(x)


if __name__ == '__main__':
    x = np.arange(64)
    print('Values: {}'.format(x))

    numProcessors = mp.cpu_count()
    print('number of cpus: {}'.format(numProcessors))

    # Set up a multi processing pool with the number less than or equal to the number of cpus available
    # Replace the value if you want to use a lower cpu count
    numProcessors = numProcessors
    pool = mp.Pool(numProcessors)
    # Use the pool map function to apply the square function to each array in the list and returns the results in a list
    squared = pool.map(square, [
                    x[numProcessors*i:numProcessors*i+numProcessors] for i in range(numProcessors)])

    print('Squared values: {}'.format(squared))