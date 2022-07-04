#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <stdlib.h>
#include <thread>
#include <ctime>
using namespace std;

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

void read_records(vector<vector<double>> &rows) {
	fstream fin;

	fin.open("neuroblastoma_CNV.csv", ios::in);
    assert(fin.is_open());

	string line, word, temp;
    bool first = true;
    int i = 0;

	while (fin >> line) {
        if (first) {
            first = false;
            continue;
        }

		stringstream s(line);

        int j = 0;
        first = true;
        rows.emplace_back();

		while (getline(s, word, ',')) {
            if (first) {
                first = false;
                continue;
            }
            double d = stod(word);
            rows.back().push_back(d);
            j++;
		}
        i++;
	}
    fin.close();
}

#ifdef CLEAR
#define HANDLE

__global__
void clear_shuffle(int n, int m, double *gpu_rows) {
    const int one_block = THREADS / WARP_SIZE;
    const int my_row = one_block * blockIdx.x + threadIdx.x / WARP_SIZE;
    if (my_row >= n) {
        return;
    }
    const int my_id = threadIdx.x % WARP_SIZE;
    double *row = gpu_rows + my_row * m;

    double sum = 0;

    for (int i = my_id; i < m; i += WARP_SIZE) {
        sum += row[i];
    }
    for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, i);
    }
    sum /= m;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        sum = __shfl_up_sync(FULL_MASK, sum, i);
    }
    double minus = sum;
    sum = 0;
    for (int i = my_id; i < m; i += WARP_SIZE) {
        double r = row[i] - minus;
        sum += r * r;
    }
    for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, i);
    }
    sum = sqrt(sum);
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        sum = __shfl_up_sync(FULL_MASK, sum, i);
    }
    for (int i = my_id; i < m; i += WARP_SIZE) {
        double r = row[i];
        row[i] = (r - minus) / sum;
    }
}

void handle_kernel(int n, int m, double *gpu_rows) {
    const int one_block = THREADS / WARP_SIZE;
    const int BLOCKS = n / one_block + (n % one_block != 0);
    auto begin = clock();
    clear_shuffle<<<BLOCKS, THREADS>>>(n, m, gpu_rows);
    cudaDeviceSynchronize();
    auto end = clock();
    auto diff = 1.0*(end - begin) / CLOCKS_PER_SEC;
    cerr << "Time needed for clear reduction: " << diff << " s" << endl;
}

#else
#define HANDLE

__device__ 
int get_my_id() {
    int res = 0;
    int tmp = threadIdx.x;
    while (tmp) {
        res++;
        tmp >>= 1;
    }
    return res;
}

__global__
void mixed_shuffle(int n, int m, double *gpu_rows) {
    const int row_id = blockIdx.x;
    const int my_id = threadIdx.x;
    double *row = gpu_rows + row_id * m;
    __shared__ double data[THREADS];
    const int last_bit = get_my_id();

    double sum = 0;
    for (int i = my_id; i < m; i += THREADS) {
        sum += row[i];
    }

    if (my_id < WARP_SIZE) {
        for (int i = THREADS / 2; i >= WARP_SIZE; i /= 2) {
            __syncthreads();
            sum += data[my_id + i];
        }

        for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, i);
        }  
        sum /= m;
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum = __shfl_up_sync(FULL_MASK, sum, i);
        }
        __syncthreads();

        for (int i = WARP_SIZE; i < THREADS; i *= 2) {
            data[my_id + i] = sum;
            __syncthreads();
        }
    }
    else {
        for (int i = THREADS / 2; i >= (1 << last_bit); i /= 2) {
            __syncthreads();
            sum += data[my_id + i];
        }
        data[my_id] = sum;
        __syncthreads();
        for (int i = 0; i < last_bit - 6; i++) {
            __syncthreads();
        }
        __syncthreads();

        for (int i = 0; i < last_bit - 6; i++) {
            __syncthreads();
        }
        __syncthreads();
        sum = data[my_id];
        for (int i = (1 << last_bit); i < THREADS; i *= 2) {
            data[my_id + i] = sum;
            __syncthreads();
        }

    }
    double minus = sum;
    sum = 0;
    for (int i = my_id; i < m; i += THREADS) {
        double r = row[i] - minus;
        sum += r * r;
    }


    if (my_id < WARP_SIZE) {
        for (int i = THREADS / 2; i >= WARP_SIZE; i /= 2) {
            __syncthreads();
            sum += data[my_id + i];
        }

        for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, i);
        }  
        sum = sqrt(sum);
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum = __shfl_up_sync(FULL_MASK, sum, i);
        }
        __syncthreads();

        for (int i = WARP_SIZE; i < THREADS; i *= 2) {
            data[my_id + i] = sum;
            __syncthreads();
        }
    }
    else {
        for (int i = THREADS / 2; i >= (1 << last_bit); i /= 2) {
            __syncthreads();
            sum += data[my_id + i];
        }
        data[my_id] = sum;
        __syncthreads();
        for (int i = 0; i < last_bit - 6; i++) {
            __syncthreads();
        }
        __syncthreads();

        for (int i = 0; i < last_bit - 6; i++) {
            __syncthreads();
        }
        __syncthreads();
        sum = data[my_id];
        for (int i = (1 << last_bit); i < THREADS; i *= 2) {
            data[my_id + i] = sum;
            __syncthreads();
        }

    }

    for (int i = my_id; i < m; i += THREADS) {
        double r = row[i];
        row[i] = (r - minus) / sum;
    }
}

void handle_kernel(int n, int m, double *gpu_rows) {
    const int BLOCKS = n;
    auto begin = clock();
    mixed_shuffle<<<BLOCKS, THREADS>>>(n, m, gpu_rows);
    cudaDeviceSynchronize();
    auto end = clock();
    auto diff = 1.0*(end - begin) / CLOCKS_PER_SEC;
    cerr << "Time needed for mixed reduction: " << diff << " s" << endl;
}

#endif

HANDLE

int main() {
    vector<vector<double>> rows;
    read_records(rows);
    
    long long n = rows.size(), m = rows[0].size();
    double *rows_arr = (double*)malloc(n * m * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            rows_arr[i * m + j] = rows[i][j];
        }
    }

    double *rows_gpu;
    if (cudaMalloc(&rows_gpu, n * m * sizeof(double)) != cudaSuccess) {
        cerr << "Error in GPU alloc 1" << endl;
        return 1;
    }
    if (cudaMemcpy(rows_gpu, rows_arr, n * m * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        cerr << "Error in GPU memcpy 1" << endl;
        return 1;
    }
    handle_kernel(n, m, rows_gpu);
    
    double *cpu_rows = (double *) malloc(n * m * sizeof(double));
    if (cudaMemcpy(cpu_rows, rows_gpu, n * m * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error in GPU memcpy 2" << endl;
        return 1;
    }
    double sum = 0, ilo = 0;
    for (int i = 0; i < m; i++) {
        sum += cpu_rows[m + i];
        ilo += cpu_rows[m + i] * cpu_rows[m + i];
    }
    cerr << "Sum: " << sum << ", ilo: " << ilo << endl;

    sum = 0, ilo = 1;
    for (int i = 0; i < n * m; i++) {
        sum += cpu_rows[i];
        ilo += cpu_rows[i] * cpu_rows[i];
    }
    ilo /= n;
    cerr << "Sum: " << sum << ", ilo: " << ilo << endl;
}
