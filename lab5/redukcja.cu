#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <stdlib.h>
#include <thread>
using namespace std;


#ifndef THREADS
#define THREADS 32
#endif
#ifndef STREAMS
#define STREAMS 1
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

__global__
void kernel1(int n, int m, double* gpu_arr) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= n) {
        return;
    }

    double sum = 0;
    for (int i = 0; i < m; i++) {
        sum += gpu_arr[i + myId * m];
    }
    sum /= m;
    for (int i = 0; i < m; i++) {
        gpu_arr[i + myId * m] -= sum;
    }
    sum = 0;
    for (int i = 0; i < m; i++) {
        double num = gpu_arr[i + myId * m];
        sum += num * num;
    }
    sum = sqrt(sum);
    for (int i = 0; i < m; i++) {
        gpu_arr[i + myId * m] /= sum;
    }
}

__device__
int my_iter(int x) {
    for (int i = 0; i <= 20; i++) {
        if (x + 1 < (1 << i)) {
            return i;
        }
    }
    assert(false);
    return -1;
}

#define min(x, y) min((long long)x, (long long)y)

#define send                                                                \
for (int i = maxHeight - 1; i > 0; i--) {                                   \
    __syncthreads();                                                        \
    if (i == height) {                                                      \
        double ile = 0;                                                     \
        ile += (dIndLeft <= mn) ? data[dIndLeft - 1] : 0;                   \
        ile += (dIndRight <= mn) ? data[dIndRight - 1] : 0;                 \
        data[threadIdx.x] += ile;                                           \
    }                                                                       \
    __syncthreads();                                                        \
}                            

#define receive                                                 \
for (int i = 1; i < maxHeight; i++) {                           \
    __syncthreads();                                            \
    if (i == height) {                                          \
        if (dIndLeft <= mn) {                                   \
            data[dIndLeft - 1] = data[threadIdx.x];             \
        }                                                       \
        if (dIndRight <= mn) {                                  \
            data[dIndRight - 1] = data[threadIdx.x];            \
        }                                                       \
    }                                                           \
    __syncthreads();                                            \
}


__global__
void kernel2(long long n, long long m, double* gpu_arr) {
    int row = blockIdx.x;
    int arr_size = (m - threadIdx.x) / THREADS + 1;
    if (arr_size == 0) {
        return;
    }
    long long mn = min(m, THREADS);

//    double *myNums = (double*)malloc(arr_size * sizeof(double));
    double *myNums = NULL;
    double sum = 0;
    int height = my_iter(threadIdx.x);
    for (int i = threadIdx.x; i < m; i += THREADS) {
        if (myNums == NULL) {
            sum += gpu_arr[m * row + i];
        }
        else {
            myNums[i / THREADS] = gpu_arr[m * row + i];
            sum += myNums[i / THREADS];
        }
    }
    

    __shared__ double data[THREADS];
    data[threadIdx.x] = sum;
    int dInd = threadIdx.x + 1;
    int dIndLeft = dInd * 2;
    int dIndRight = dInd * 2 + 1;
    int maxHeight = my_iter(mn-1);

    send;
    receive;

    sum = data[threadIdx.x] / m;
    double ilo = 0;
    for (int i = threadIdx.x; i < m; i += THREADS) {
        if (myNums == NULL) {
            gpu_arr[row * m + i] -= sum;
            ilo += gpu_arr[row * m + i] * gpu_arr[row * m + i];
        }
        else {
            int nr = i / THREADS;
            myNums[nr] -= sum;
            ilo += myNums[nr] * myNums[nr];
        }
    }
    data[threadIdx.x] = ilo;
    send;
    receive;
    
    ilo = sqrt(data[threadIdx.x]);

    for (int i = threadIdx.x; i < m; i += THREADS) {
        if (myNums == NULL) {
            gpu_arr[m * row + i] /= ilo;
        }
        else {
            gpu_arr[m * row + i] = myNums[i / THREADS] / ilo;
//          gpu_arr[m * row + i] = ilo2;
        }
    }
    free(myNums);
}





__global__
void dot_product(long long a, long long b, long long n, long long m, double *gpu_arr, double *res) {
    if (a >= n || b >= n) {
        return;
    }
    if (a == b && threadIdx.x == 0) {
        res[n * a + b] = 1;
        return;
    }
    if (a == b) {
        return;
    }
    if (a > b) {
        return;
    }

    int arr_size = (m - threadIdx.x) / THREADS + 1;
    if (arr_size == 0) {
        return;
    }
    long long mn = min(m, THREADS);

    double sum = 0;
    int height = my_iter(threadIdx.x);
    for (int i = threadIdx.x; i < m; i += THREADS) {
        sum += gpu_arr[m * a + i] * gpu_arr[m * b + i];
    }
    

    __shared__ double data[THREADS];
    data[threadIdx.x] = sum;
    int dInd = threadIdx.x + 1;
    int dIndLeft = dInd * 2;
    int dIndRight = dInd * 2 + 1;
    int maxHeight = my_iter(mn-1);

    send;
    if (threadIdx.x > 0) {
        return;
    }
    
    res[a * n + b] = data[0];
    res[b * n + a] = data[0];
}


__global__
void dot_product2(long long n, long long m, double *gpu_arr, double *res) {
    long long a = blockIdx.x;
    long long b = blockIdx.y;
    if (a >= n || b >= n) {
        return;
    }
    if (a == b && threadIdx.x == 0) {
        res[n * a + b] = 1;
        return;
    }
    if (a == b) {
        return;
    }
    if (a > b) {
        return;
    }

    int arr_size = (m - threadIdx.x) / THREADS + 1;
    if (arr_size == 0) {
        return;
    }
    long long mn = min(m, THREADS);

    double sum = 0;
    int height = my_iter(threadIdx.x);
    for (int i = threadIdx.x; i < m; i += THREADS) {
        sum += gpu_arr[m * a + i] * gpu_arr[m * b + i];
    }
    

    __shared__ double data[THREADS];
    data[threadIdx.x] = sum;
    int dInd = threadIdx.x + 1;
    int dIndLeft = dInd * 2;
    int dIndRight = dInd * 2 + 1;
    int maxHeight = my_iter(mn-1);

    send;
    if (threadIdx.x > 0) {
        return;
    }
    
    res[a * n + b] = data[0];
    res[b * n + a] = data[0];
}



int main() {
    cout << "Streams: " << STREAMS << ", Threads: " << THREADS << endl; 
    vector<vector<double>> rows;
    read_records(rows);
    
//    long long m = rows.size(), n = rows[0].size();
    long long n = rows.size(), m = rows[0].size();
    double *rows_arr = (double*)malloc(n * m * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            rows_arr[i * m + j] = rows[i][j];
        }
    }
    double* gpu_rows;

    if (cudaMalloc(&gpu_rows, n * m * sizeof(double))) {
        cout << "Error in allocation 0" << endl;
        return 1;
    }
    if (cudaMemcpy(gpu_rows, rows_arr, n * m * sizeof(double), cudaMemcpyHostToDevice)) {
        cout << "Error in copy 1" << endl;
        return 1; 
    }
    free(rows_arr);

    auto start = clock();

//    int BLOCKS = n / THREADS + 1;
 //   kernel1<<<BLOCKS, THREADS>>>(n, m, gpu_rows);

    int BLOCKS = n;
    kernel2<<<BLOCKS, THREADS>>>(n, m, gpu_rows);

    cudaDeviceSynchronize();
    
    auto end = clock();
    auto diff = 1.0*(end - start)/CLOCKS_PER_SEC;
    printf("Computing took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);

    double *wiersze;
    wiersze = (double*)malloc(n * m * sizeof(double));
    int err;
    if (err = cudaMemcpy(wiersze, gpu_rows, n * m * sizeof(double), cudaMemcpyDeviceToHost)) {
        cout << "Error in copy " << err << " " << n << " " << m << " " << n * m * sizeof(double) << endl;
        return 1;
    }

    double sum = 0, ilo = 0;
    long long k = n-1;
    for (int i = 0; i < m; i++) {
        sum += wiersze[k * m + i];
        ilo += wiersze[k * m + i] * wiersze[k * m + i];
    }
    double *dot_gpu;
    if (cudaMalloc(&dot_gpu, n * n * sizeof(double))) {
        cout << "Error in allocation 1" << endl;
        return 1;
    }
    cudaStream_t streams[STREAMS];
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(streams + i);
    }

    start = clock();
/*    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int ind = ((2*n-i+1)*i/2 + (j - i)) % STREAMS;
            dot_product<<<1, THREADS, 0, streams[ind]>>>(i, j, n, m, gpu_rows, dot_gpu);
        }
    }*/
    dim3 blocks(n, n, 1);
    dot_product2<<<blocks, THREADS>>>(n, m, gpu_rows, dot_gpu);
    cudaDeviceSynchronize();
    end = clock();
    diff = 1.0*(end - start)/CLOCKS_PER_SEC;
    printf("Computing dot product took %lf s\n\n", diff);
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    double *dot_cpu = (double*)malloc(n * n * sizeof(double));
    int err2;
    if (err2 = cudaMemcpy(dot_cpu, dot_gpu, n * n * sizeof(double), cudaMemcpyDeviceToHost)) {
        cout << "Error in copy " << err2 << " " << n << " " << m << " " << n * n * sizeof(double) << endl;
        return 1;
    }
 //   for (int i = 0; i < n; i++) {
 //       cout << dot_cpu[4*n + i] << endl;
//    }



    if (cudaFree(gpu_rows)) {
        cout << "Error in free" << endl;
        return 1;
    }
    free(wiersze);

    


    m = rows.size(), n = rows[0].size();
    rows_arr = (double*)malloc(n * m * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            rows_arr[j * m + i] = rows[i][j];
        }
    }

    if (cudaMalloc(&gpu_rows, n * m * sizeof(double))) {
        cout << "Error in allocation 2" << endl;
        return 1;
    }
    if (cudaMemcpy(gpu_rows, rows_arr, n * m * sizeof(double), cudaMemcpyHostToDevice)) {
        cout << "Error in copy 1" << endl;
        return 1; 
    }
    free(rows_arr);

    start = clock();

//    int BLOCKS = n / THREADS + 1;
 //   kernel1<<<BLOCKS, THREADS>>>(n, m, gpu_rows);

    BLOCKS = n;
    kernel2<<<BLOCKS, THREADS>>>(n, m, gpu_rows);

    cudaDeviceSynchronize();
    
    end = clock();
    diff = 1.0*(end - start)/CLOCKS_PER_SEC;
    printf("Computing took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);


    double *kolumny;
    kolumny = (double*)malloc(n * m * sizeof(double));
    if (err = cudaMemcpy(kolumny, gpu_rows, n * m * sizeof(double), cudaMemcpyDeviceToHost)) {
        cout << "Error in copy " << err << " " << n << " " << m << " " << n * m * sizeof(double) << endl;
        return 1;
    }




    if (cudaFree(gpu_rows)) {
        cout << "Error in free" << endl;
        return 1;
    }
    free(kolumny);
}