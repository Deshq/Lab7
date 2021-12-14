from random import randint


def get_matrix(size):
    
    return [[randint(0, 9) for i in range(size)] \
        for j in range(size)]
