import math


# Retrieves the nth of the Pascal's triangle(gaussian discrete approximation)
# DON'T USE this, it's not as efficient as the convolutional way
def binomial(L):
    return [math.factorial(L) / (math.factorial(x) * math.factorial(L - x)) for x in range(L + 1)]


L = input('Give the Level value: ')
print(binomial(L))
print
