import ray

# Initialize Ray
ray.init()

# Task for printing the statement
@ray.remote
def func():
    return "This is my first program using Ray"

# Execute the task
future = func.remote()

# Retrieve the result
result = ray.get(future)
print(result)  

# Shutdown Ray
ray.shutdown()
