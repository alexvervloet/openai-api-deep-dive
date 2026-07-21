"""A small snippet to point `ask.py` at. It has a subtle bug or two on purpose 
ask the model to find them!"""


def average(numbers):
    total = 0
    for n in numbers:
        total += n
    # Bug: integer division, and no guard against an empty list.
    return total // len(numbers)


def running_max(values):
    best = 0  # Bug: assumes all values are non-negative.
    for v in values:
        if v > best:
            best = v
    return best


if __name__ == "__main__":
    print(average([1, 2, 2]))        # expect ~1.67, prints 1
    print(running_max([-5, -3, -8])) # expect -3, prints 0
