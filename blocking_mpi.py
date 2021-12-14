import sys
from mpi4py import MPI

from app.config import MATRIX_DIMENSION

from utils.matrix_multiplication import matrix_multiplication
from utils.init_matrix import get_matrix
from utils.split_matrix import split_matrix

from random import randint

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1

TaskMaster = 0

MatrixDimension = int(MATRIX_DIMENSION)

MatrixA = []
MatrixB = []

ResultMatrix = []


def init_matrix():

    global MatrixA
    MatrixA = get_matrix(MatrixDimension)

    global MatrixB
    MatrixB = get_matrix(MatrixDimension)


def distribute_and_send_data():

    rows = split_matrix(MatrixA, workers)
    process_id = 1

    for row in rows:

        comm.send(row, dest=process_id, tag=1)
        comm.send(MatrixB, dest=process_id, tag=2)
        process_id = process_id + 1


def assemble_matrix_data():

    global ResultMatrix
    
    process_id = 1

    for n in range(workers):

        row = comm.recv(source=process_id, tag=process_id)
        ResultMatrix += row
        process_id = process_id + 1


def master_operation():

    t_start = MPI.Wtime()

    distribute_and_send_data()
    assemble_matrix_data()

    t_diff = MPI.Wtime() - t_start
    
    print("[master] Process %d finished in %5.4fs.\n" %(rank, t_diff))


def slave_operation():

    t_start = MPI.Wtime()

    # receive data from master node
    row = comm.recv(source=TaskMaster, tag=1)
    mtrx = comm.recv(source=TaskMaster, tag=2)
   
    # multiply the received matrix and send the result back to master
    result = matrix_multiplication(row, mtrx) 
    comm.send(result, dest=TaskMaster, tag=rank)
    
    t_diff = MPI.Wtime() - t_start

    print("[slave] Process %d finished in %5.4fs.\n" %(rank, t_diff))


if __name__ == '__main__':

    if rank == TaskMaster:

        init_matrix()

        master_operation()
        
    else:
        slave_operation()
