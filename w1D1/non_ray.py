import time

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

number = 150
start_time = time.time()
factorial_result = factorial(number)
end_time = time.time()

compute_time = end_time - start_time
print(f"Program without Ray\n")
print(f"Number: 150")
print(f"Factorial of {number} is {factorial_result}")
print(f"Compute time: {compute_time} seconds")
