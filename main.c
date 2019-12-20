#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define sz 945


int main(int argc, char *argv[]) {
//    MPI_COMM_WORLD MPI_COMM_WORLD;
    MPI_Status status[4];
    int rank, size;
    int i, j, k;
    int dims[2];
    int *a, *b, *c;
    int *a_all, *b_all, *c_all;
    double tm;
    int sz_block;
    int ndims = 2;
    a_all = (int *) malloc(sz * sz * sizeof(int));
    b_all = (int *) malloc(sz * sz * sizeof(int));
    c_all = (int *) calloc(sz * sz, sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request req[4];

    dims[0] = 0;
    dims[1] = 0;
    MPI_Dims_create(size, ndims, dims);

    if (dims[0] != dims[1]) {
        if (rank == 0) printf("The number of processors must be a square.\n");
        MPI_Finalize();
        return 0;
    }

    sz_block = sz / dims[0];
    a = (int *) malloc(sz_block * sz_block * sizeof(int));
    b = (int *) malloc(sz_block * sz_block * sizeof(int));
    c = (int *) calloc(sz_block * sz_block, sizeof(int));

    if (rank == 0) printf("MPI processors: %d\n", size);
    int i_coord = rank / dims[0];
    int j_coord = rank % dims[0];
    for (i = 0; i < sz_block; i++) {
        for (j = 0; j < sz_block; j++) {
            a[i * sz_block + j] = rand() % 10000;
            a_all[(i + i_coord * sz_block) * sz + (j_coord * sz_block + j)] = a[i * sz_block + j];
            b[i * sz_block + j] = rand() % 10000;
            b_all[(i + i_coord * sz_block) * sz + (j_coord * sz_block + j)] = b[i * sz_block + j];
            c[i * sz_block + j] = 0;
            c_all[(i + i_coord * sz_block) * sz + (j_coord * sz_block + j)] = 0;
        }
    }
    if (rank == 0) {
        int *a_tmp = (int *) malloc(sz_block * sz_block * sizeof(int));
        for (int proc = 1; proc < size; ++proc) {
            MPI_Recv(a_tmp, sz_block * sz_block, MPI_INT, proc, 2020, MPI_COMM_WORLD, &status[0]);
            int i_coord_tmp = proc / dims[0];
            int j_coord_tmp = proc % dims[0];
            for (i = 0; i < sz_block; ++i) {
                for (j = 0; j < sz_block; ++j) {
                    a_all[(i + i_coord_tmp * sz_block) * sz + (j_coord_tmp * sz_block + j)] = a_tmp[i * sz_block + j];
                }
            }
        }
        free(a_tmp);
    } else {
        MPI_Send(a, sz_block * sz_block, MPI_INT, 0, 2020,
                 MPI_COMM_WORLD);
    }
    if (rank == 0) {
        int *b_tmp = (int *) malloc(sz_block * sz_block * sizeof(int));
        for (int proc = 1; proc < size; ++proc) {
            MPI_Recv(b_tmp, sz_block * sz_block, MPI_INT, proc, 2020, MPI_COMM_WORLD, &status[0]);
            int i_coord_tmp = proc / dims[0];
            int j_coord_tmp = proc % dims[0];
            for (i = 0; i < sz_block; ++i) {
                for (j = 0; j < sz_block; ++j) {
                    b_all[(i + i_coord_tmp * sz_block) * sz + (j_coord_tmp * sz_block + j)] = b_tmp[i * sz_block + j];
                }
            }
        }
        free(b_tmp);
    } else {
        MPI_Send(b, sz_block * sz_block, MPI_INT, 0, 2020,
                 MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tm = MPI_Wtime();
    int *recv_tmp_a, *recv_tmp_b;
    // начальная пересылка
    MPI_Isend(a, sz_block * sz_block, MPI_INT, i_coord * dims[0] + (j_coord - i_coord + dims[0]) % dims[0], 2019,
              MPI_COMM_WORLD, &req[0]);

    MPI_Isend(b, sz_block * sz_block, MPI_INT, ((i_coord - j_coord + dims[0]) % dims[0]) * dims[0] + j_coord, 2019,
              MPI_COMM_WORLD, &req[1]);
    recv_tmp_a = (int *) malloc(sz_block * sz_block * sizeof(int));
    recv_tmp_b = (int *) malloc(sz_block * sz_block * sizeof(int));

    MPI_Irecv(recv_tmp_a, sz_block * sz_block, MPI_INT, i_coord * dims[0] + (j_coord + i_coord) % dims[0], 2019,
              MPI_COMM_WORLD, &req[2]);
    MPI_Irecv(recv_tmp_b, sz_block * sz_block, MPI_INT, ((i_coord + j_coord + dims[0]) % dims[0]) * dims[0] + j_coord,
              2019,
              MPI_COMM_WORLD, &req[3]);

    MPI_Waitall(4, &req[0], &status[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    int *sw_a, *sw_b;
    sw_a = recv_tmp_a;
    recv_tmp_a = a;
    a = sw_a;
    sw_a = NULL;
    sw_b = recv_tmp_b;
    recv_tmp_b = b;
    b = sw_b;
    sw_b = NULL;

    for (int iteration = 0; iteration < dims[0]; ++iteration) {
        //умножение
        for (i = 0; i < sz_block; i++) {
            for (k = 0; k < sz_block; k++) {
                for (j = 0; j < sz_block; j++) {
                    c[i * sz_block + j] += a[i * sz_block + k] * b[k * sz_block + j];
                }
            }
        }
        MPI_Isend(a, sz_block * sz_block, MPI_INT, i_coord * dims[0] + (j_coord - 1 + dims[0]) % dims[0],
                  2019 + iteration,
                  MPI_COMM_WORLD, &req[0]);

        MPI_Isend(b, sz_block * sz_block, MPI_INT, ((i_coord - 1 + dims[0]) % dims[0]) * dims[0] + j_coord,
                  2019 + iteration,
                  MPI_COMM_WORLD, &req[1]);

        MPI_Irecv(recv_tmp_a, sz_block * sz_block, MPI_INT, i_coord * dims[0] + (j_coord + 1) % dims[0],
                  2019 + iteration,
                  MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(recv_tmp_b, sz_block * sz_block, MPI_INT, ((i_coord + 1 + dims[0]) % dims[0]) * dims[0] + j_coord,
                  2019 + iteration,
                  MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(4, &req[0], &status[0]);
        sw_a = recv_tmp_a;
        recv_tmp_a = a;
        a = sw_a;
        sw_a = NULL;
        sw_b = recv_tmp_b;
        recv_tmp_b = b;
        b = sw_b;
        sw_b = NULL;
    }


    if (rank == 0) {
        for (i = 0; i < sz_block; ++i) {
            for (j = 0; j < sz_block; ++j) {
                c_all[(i + i_coord * sz_block) * sz + (j_coord * sz_block + j)] = c[i * sz_block + j];
            }
        }
        int *c_tmp = (int *) malloc(sz_block * sz_block * sizeof(int));
        for (int proc = 1; proc < size; ++proc) {
            MPI_Recv(c_tmp, sz_block * sz_block, MPI_INT, proc, 2020, MPI_COMM_WORLD, &status[0]);
            int i_coord_tmp = proc / dims[0];
            int j_coord_tmp = proc % dims[0];
            for (i = 0; i < sz_block; ++i) {
                for (j = 0; j < sz_block; ++j) {
                    c_all[(i + i_coord_tmp * sz_block) * sz + (j_coord_tmp * sz_block + j)] = c_tmp[i * sz_block + j];
                }
            }
        }
        free(c_tmp);
    } else {
        MPI_Send(c, sz_block * sz_block, MPI_INT, 0, 2020,
                 MPI_COMM_WORLD);
    }

    free(recv_tmp_a);
    free(recv_tmp_b);

    MPI_Barrier(MPI_COMM_WORLD);
    tm = MPI_Wtime() - tm;
    free(a);
    free(b);
    free(c);
    if (rank == 0)
        printf("CPU time: %lf\n", tm);


//    if (rank == 0) {
//        for (i = 0; i < sz; ++i) {
//            for (k = 0; k < sz; ++k) {
//                for (j = 0; j < sz; ++j) {
//                    c_all[i * sz_block + j] -= a_all[i * sz_block + k] * b_all[k * sz_block + j];
//                }
//            }
//        }
//        for (i = 0; i < sz; ++i) {
//            for (j = 0; j < sz; ++j) {
//                printf("%d ", c_all[i * sz + j]);
//            }
//        }
//      }
    MPI_Finalize();
    return 0;
}
