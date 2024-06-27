import ray
import time

# Ray initialization
ray.init()

# Ray task definition for multiplying numbers
@ray.remote
def product(start, end):
    res = 1
    for i in range(start, end + 1):
        res *= i
    return res

# Ray actor definition for factorial
@ray.remote
class Factorial:
    def __init__(self):
        self.intermediate_results = []

    def calculate_factorial(self, n):
        # Define ranges for parallel computation
        num_tasks = 4
        step = n // num_tasks
        ranges = [(i * step + 1, (i + 1) * step) for i in range(num_tasks)]
        ranges[-1] = (ranges[-1][0], n)  # Adjust the last range

        # Launch parallel tasks
        futures = [product.remote(start, end) for start, end in ranges]

        # Retrieve and multiply intermediate results
        self.intermediate_results = ray.get(futures)
        factorial_result = 1
        for result in self.intermediate_results:
            factorial_result *= result

        return factorial_result

# Create an instance of the Factorial actor
calculator = Factorial.remote()

# Calculate the factorial of a number and measure the compute time
number = 150
start_time = time.time()
factorial = ray.get(calculator.calculate_factorial.remote(number))
end_time = time.time()

compute_time = end_time - start_time
print(f"Program without Ray\n")
print(f"Number: 150\n")
print(f"Factorial: {factorial}")
print(f"Compute time: {compute_time} seconds")

# Shutdown Ray
ray.shutdown()

