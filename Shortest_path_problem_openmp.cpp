#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <time.h>
#include<Windows.h>
#include <stdio.h>
#include "omp.h"

using namespace std;
#define N 5// vertices
#define INFINITY 1000000
#define no_of_threads 4
int* Distance;
    //translate 2-dimension coordinate to 1-dimension
    int convert_dimension_to_2D_from_1D(int x, int y, int n) {
        return x * n + y;
    }
    int print_result(bool has_negative_cycle, int* Distance) {
        if (!has_negative_cycle) {
            for (int i = 0; i < N; i++) {
                if (Distance[i] > INFINITY)
                    Distance[i] = INFINITY;
                cout<< Distance[i] << "\n";
            }
        }
        else {
           cout<<"FOUND NEGATIVE CYCLE!" << endl;
        }
        return 0;
    }
void Bellman_Ford(int n, int mat[N], int* dist, bool* has_negative_cycle) {

    int local_start[no_of_threads], local_end[no_of_threads];
    *has_negative_cycle = false;

    //step 1: set openmp thread number
    omp_set_num_threads(no_of_threads);

    //step 2: find local task range
    int ave = n / no_of_threads;
#pragma omp parallel for
    for (int i = 0; i < no_of_threads; i++) {
        local_start[i] = ave * i;
        local_end[i] = ave * (i + 1);
        if (i == no_of_threads - 1) {
            local_end[i] = n;
        }
    }

    //initialize distances
    //bellmanford algo apply here
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dist[i] = INFINITY;
    }
    //root vertex always has distance 0
    dist[0] = 0;
    int iter_num = 0;
    bool has_change;
    bool local_has_change[no_of_threads];
#pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        //bellman-ford algorithm
        for (int iter = 0; iter < n - 1; iter++) {
            local_has_change[my_rank] = false;
            for (int u = 0; u < n; u++) {
                for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {
                    int weight = mat[convert_dimension_to_2D_from_1D(u, v, n)];
                    if (weight < INFINITY) {
                        int new_dis = dist[u] + weight;
                        if (new_dis < dist[v]) {
                            local_has_change[my_rank] = true;
                            dist[v] = new_dis;
                        }
                    }
                }
            }
#pragma omp barrier
#pragma omp single
            {
                iter_num++;
                has_change = false;
                for (int rank = 0; rank < no_of_threads; rank++) {
                    has_change |= local_has_change[rank];
                }
            }
            if (!has_change) {
                break;
            }
        }
    }

    //do one more iteration to check negative cycles
    if (iter_num == n - 1) {
        has_change = false;
        for (int u = 0; u < n; u++) {
#pragma omp parallel for reduction(|:has_change)
            for (int v = 0; v < n; v++) {
                int weight = mat[u * n + v];
                if (weight < INFINITY) {
                    if (dist[u] + weight < dist[v]) { // if we can relax one more step, then we find a negative cycle
                        has_change = true;;
                    }
                }
            }
        }
        *has_negative_cycle = has_change;
    }
}

int main(int argc, char** argv) {
   int Matrix[25] = { 0, -1 , 4 , 1000000 ,1000000 , 1000000 , 0 , 3 ,2,2,1000000 ,1000000, 0 , 1000000, 1000000 , 1000000,1 ,5,0,1000000 ,1000000,1000000,1000000,-3 , 0 };
    bool has_negative_cycle = false;
    Distance = (int*)malloc(sizeof(int) *N);
    double time1 = omp_get_wtime();
    //bellman-ford algorithm
    Bellman_Ford( N, Matrix, Distance, &has_negative_cycle);
    double time2 = omp_get_wtime();
    cout<<setprecision(6) << "Time(s): " << (time2-time1) << endl;
    print_result(has_negative_cycle, Distance);
    free(Distance);
    return 0;
}