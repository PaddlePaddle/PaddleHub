#!/usr/bin/env python
# -*- coding: utf-8 -*-


class MPIHelper(object):
    def __init__(self):
        try:
            from mpi4py import MPI
        except:
            # local run
            self._size = 1
            self._rank = 0
            self._multi_machine = False

            import socket
            self._name = socket.gethostname()
        else:
            # in mpi environment
            self._comm = MPI.COMM_WORLD
            self._size = self._comm.Get_size()
            self._rank = self._comm.Get_rank()
            self._name = MPI.Get_processor_name()
            if self._size > 1:
                self._multi_machine = True
            else:
                self._multi_machine = False

    @property
    def multi_machine(self):
        return self._multi_machine

    @property
    def rank(self):
        return self._rank

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return self._name

    def bcast(self, data):
        if self._multi_machine:
            # call real bcast
            return self._comm.bcast(data, root=0)
        else:
            # do nothing
            return data

    def gather(self, data):
        if self._multi_machine:
            # call real gather
            return self._comm.gather(data, root=0)
        else:
            # do nothing
            return [data]

    def allgather(self, data):
        if self._multi_machine:
            # call real allgather
            return self._comm.allgather(data)
        else:
            # do nothing
            return [data]

    # calculate split range on mpi environment
    def split_range(self, array_length):
        if self._size == 1:
            return 0, array_length
        average_count = array_length // self._size
        if array_length % self._size == 0:
            return average_count * self._rank, average_count * (self._rank + 1)
        else:
            if self._rank < array_length % self._size:
                return (average_count + 1) * self._rank, (average_count + 1) * (
                    self._rank + 1)
            else:
                start = (average_count + 1) * (array_length % self._size) \
                      + average_count * (self._rank - array_length % self._size)
                return start, start + average_count


if __name__ == "__main__":

    mpi = MPIHelper()
    print("Hello world from process {} of {} at {}.".format(
        mpi.rank, mpi.size, mpi.name))

    all_node_names = mpi.gather(mpi.name)
    print("all node names using gather: {}".format(all_node_names))

    all_node_names = mpi.allgather(mpi.name)
    print("all node names using allgather: {}".format(all_node_names))

    if mpi.rank == 0:
        data = range(10)
    else:
        data = None
    data = mpi.bcast(data)
    print("after bcast, process {} have data {}".format(mpi.rank, data))

    data = [i + mpi.rank for i in data]
    print("after modify, process {} have data {}".format(mpi.rank, data))

    new_data = mpi.gather(data)
    print("after gather, process {} have data {}".format(mpi.rank, new_data))

    # test for split
    for i in range(12):
        length = i + mpi.size  # length should >= mpi.size
        [start, end] = mpi.split_range(length)
        split_result = mpi.gather([start, end])
        print("length {}, split_result {}".format(length, split_result))
