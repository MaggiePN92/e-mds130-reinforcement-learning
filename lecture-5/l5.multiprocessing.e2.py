import multiprocessing as mp
import numpy as np


# Takes an array and squares each element
def square(i, x, queue):
    print('in process: {}'.format(i))
    queue.put(np.square(x)) # We store the squared value of each such into an queue


if __name__ == '__main__':
    processes = [] # A list to store the reference to each process    
    queue = mp.Queue() # A multiprocessing queue. A data structure that is shared across all processes
    
    x = np.arange(64)
    print('Values: {}'.format(x))

    numProcessors = mp.cpu_count()
    print('number of cpus: {}'.format(numProcessors))

    # Set up a multi processing pool with the number less than or equal to the number of cpus available
    # Replace the value if you want to use a lower cpu count
    numProcessors = numProcessors

    # Starts as many processes as specified in numProcessors. Each process performs the square function 
    # with the process index, data chunk and queue as arguments
    for i in range(numProcessors):
        startIndex = numProcessors*i
        process = mp.Process(target=square, args=(i, x[startIndex:startIndex+numProcessors], queue))
        process.start()
        processes.append(process)
    
    # Executes after all the processes have completed
    for process in processes:
        process.join() # Wait return until all processes have completed
    
    # Terminates each process
    for process in processes:
        process.terminate()

    results = []
    while not queue.empty():
        results.append(queue.get()) # Pops a result sequence from the queue and adds it to the results list
    
    print('Squared values: {}'.format(results))
