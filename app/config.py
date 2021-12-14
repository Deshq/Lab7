from environs import Env

env = Env()
env.read_env()


MATRIX_DIMENSION = env.str('MATRIX_DIMENSION')

COUNT_PROCESSES = env.str('COUNT_PROCESSES')