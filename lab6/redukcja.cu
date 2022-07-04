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


#ifndef THREADS
#define THREADS 32
#endif

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

__device__
void get_my_vectors(int n, int m, int *x, int *y) {
    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    int row = myId / (n + 1);
    if (myId % (n + 1) > row) {
        *x = row;
        *y = myId % (n + 1) - 1;
    }
    else {
        *x = n - row - 1;
        *y = *x + myId % (n + 1);
    }
}




#ifndef THREADSX 
#define HANDLE

__global__
void one_thread_one_product(int n, int m, double *gpu_arr, double *gpu_res) {
    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    if (myId >= n * (n + 1) / 2) {
        return;
    }
    int x, y;
    get_my_vectors(n, m, &x, &y);
    double *vec1 = gpu_arr + m * x;
    double *vec2 = gpu_arr + m * y;
    double res = 0;
    for (int i = 0; i < m; i++) {
        res += vec1[i] * vec2[i];
    }
    gpu_res[n * x + y] = res;
    gpu_res[n * y + x] = res;
}

void handle_gpu(int n, int m, double *rows_gpu, double *gpu_res) {
    int blocks = n * (n + 1) / 2 / THREADS + 1;
    cudaDeviceSynchronize();
    auto begin = clock();
    one_thread_one_product<<<blocks, THREADS>>>(n, m, rows_gpu, gpu_res);
    cudaDeviceSynchronize();
    auto end = clock();
    auto diff = 1.0*(end - begin) / CLOCKS_PER_SEC;
    cerr << "Time needed for " << THREADS << " 1d threads: " << diff << " s" << endl;
}

#elif K 
#define HANDLE

inline __device__ unsigned get_lane_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

inline __device__ unsigned get_warp_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
  return ret;
}

__global__
void one_thread_one_part_2d(int n, int m, int k, double *gpu_arr, double *gpu_res) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // faktyczny x
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // faktyczny y
    const int all_threads = gridDim.x * blockDim.x * gridDim.y * blockDim.y;
    const int id = y * blockDim.x + x;    // faktyczne id
    const int len = min(K, m - k * K);
    extern __shared__ double vec[];
    for (int i = id; i < n * len; i += all_threads) {
        int line = i / len;
        int num = i % len;
        vec[i] = gpu_arr[line * m + k * K + num];
    }
    if (x >= n || y >= n || x < y) {
        return;
    }
    double *vec1 = vec + len * x;
    double *vec2 = vec + len * y;
    double res = 0;
    for (int i = 0; i < len; i++) {
        res += vec1[i] * vec2[i];
    }
    int num = y * (y + 1) / 2 + x;
    gpu_res[num] = res;
}

__global__
void sum_parts(int n, int m, double **gpu_arr, double *gpu_res) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // faktyczny x
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // faktyczny y

    if (x >= n || y >= n || x < y) {
        return;
    }
    double res = 0;
    int num = y * (y + 1) / 2 + x;
    for (int i = 0; i < m; i++) {
        res += gpu_arr[i][num];
    }
    gpu_res[y * n + x] = res;
    gpu_res[x * n + y] = res;
}

void handle_gpu(int n, int m, double *rows_gpu, double *gpu_res) {
    auto begin = clock();
    const int STREAMS = m / K + (m % K != 0);
    cudaStream_t *streams = (cudaStream_t *) malloc(STREAMS * sizeof(cudaStream_t));
    double **part_res = (double **) malloc(STREAMS * sizeof(double *));
    double **part_res_gpu;
    if (cudaMalloc(&part_res_gpu, STREAMS * sizeof(double *)) != cudaSuccess) {
        cerr << "Error in GPU alloc 3" << endl;
        return;
    }
    cudaStream_t cpy_stream;
    if (cudaStreamCreate(&cpy_stream) != cudaSuccess) {
        cerr << "Error in GPU stream create 0" << endl;
        return;
    }

    dim3 threads(THREADSX, THREADSY);
    dim3 blocks(n / THREADSX + (n % THREADSX != 0), n / THREADSY + (n % THREADSY != 0));
    for (int i = 0; i < STREAMS; i++) {
        if (cudaStreamCreate(&streams[i]) != cudaSuccess) {
            cerr << "Error in GPU stream create " << i + 1 << endl;
            return;
        }
        if (cudaMalloc(&part_res[i], n * (n + 1) / 2 * sizeof(double)) != cudaSuccess) {
            cerr << "Error in GPU alloc " << i + 4 << endl;
            return;
        }
        int len = min(K, m - i * K);
        one_thread_one_part_2d<<<blocks, threads, len * n * sizeof(double), streams[i]>>>(n, m, i, rows_gpu, part_res[i]);
    }
    if (cudaMemcpyAsync((void *) part_res_gpu, (void *) part_res, STREAMS * sizeof(double *), cudaMemcpyHostToDevice, cpy_stream) != cudaSuccess) {
        cerr << "Error in GPU async copy 0" << endl;
        return;
    }
    cudaDeviceSynchronize();
    // double *cpu_part = (double *) malloc(n * (n + 1) / 2 * sizeof(double));
    // if (cudaMemcpy(cpu_part, part_res[1], n * (n + 1) / 2 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
    //     cerr << "memcpy z debuga" << endl;
    //     return;
    // }
    // cerr << "DEBUG " << STREAMS << endl;
    // for (int i = 0; i < 10; i++) {
    //     cerr << cpu_part[i] << endl; 
    // }

    sum_parts<<<blocks, threads>>>(n, STREAMS, part_res_gpu, gpu_res);
    cudaStreamDestroy(cpy_stream);
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaDeviceSynchronize();
    auto end = clock();
    auto diff = 1.0*(end - begin) / CLOCKS_PER_SEC;
    cerr << "Time needed for (" << THREADSX << ", " << THREADSY << ")" << " 2d threads: " << diff << " s" << endl;

    if (cudaFree(part_res_gpu) != cudaSuccess) {
        cerr << "Error in GPU free 0" << endl;
        return;
    }
    for (int i = 0; i < STREAMS; i++) {
        if (cudaFree(part_res[i]) != cudaSuccess) {
            cerr << "Error in GPU free " << i + 1 << endl;
            return;
        }
    }
}

#else
#define HANDLE

__global__
void one_thread_one_product_2d(int n, int m, double *gpu_arr, double *gpu_res) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n || x < y) {
        return;
    }
    double *vec1 = gpu_arr + m * x;
    double *vec2 = gpu_arr + m * y;
    double res = 0;
    for (int i = 0; i < m; i++) {
        res += vec1[i] * vec2[i];
    }
    gpu_res[n * x + y] = res;
    gpu_res[n * y + x] = res;
}

void handle_gpu(int n, int m, double *rows_gpu, double *gpu_res) {
    int blocks_x = n / THREADSX + 1;
    int blocks_y = n / THREADSY + 1;
    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(THREADSX, THREADSY);
    cudaDeviceSynchronize();
    auto begin = clock();
    one_thread_one_product_2d<<<blocks, threads>>>(n, m, rows_gpu, gpu_res);
    cudaDeviceSynchronize();
    auto end = clock();
    auto diff = 1.0*(end - begin) / CLOCKS_PER_SEC;
    cerr << "Time needed for (" << THREADSX << ", " << THREADSY << ")" << " 2d threads: " << diff << " s" << endl;
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

    double *gpu_res;
    if (cudaMalloc(&gpu_res, n * n * sizeof(double)) != cudaSuccess) {
        cerr << "Error in GPU alloc 2" << endl;
        return 1;
    }
    handle_gpu(n, m, rows_gpu, gpu_res);


    double *cpu_res = (double *)malloc(n * n * sizeof(double));
    if (cpu_res == nullptr) {
        cerr << "Error in CPU malloc 1" << endl;
        return 1;
    }
    if (cudaMemcpy(cpu_res, gpu_res, n * n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error in GPU memcpy 2 " << cudaGetErrorName(cudaGetLastError()) << endl;
        return 1;
    }
    cudaDeviceSynchronize();

    // int k = 5;
    // for (int i = 0; i < k; i++) {
    //     for (int j = 0; j < k; j++) {
    //         cerr << "(" << i << ", " << j << ") -> " << cpu_res[i * n + j] << endl;
    //     }
    // }

    if (cudaFree(gpu_res) != cudaSuccess) {
        cerr << "Error in GPU free 1" << endl;
    }
    if (cudaFree(rows_gpu) != cudaSuccess) {
        cerr << "Error in GPU free 2" << endl;
    }
    free(cpu_res);
}
