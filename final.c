/*
 ============================================================================
 Name        : LU.c
 Author      : Ronak Bandwal , Sahej Ganeriwala
 Description : LU Factorization using OpenMP : study of scalability
 ============================================================================
 */
#include<stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Dynamically allocates a 2D-matrix of size nxn
double **make2dmatrix(long n) {
	long i;
	double **m;
	m = (double**)malloc(n*sizeof(double*));
	for(i=0;i<n;i++)
		m[i] = (double*)malloc(n*sizeof(double));
	return m;
}

// Prints the given 2D-matrix
void printmatrix(double **A, long n) {
	long i, j;
	for (i=0;i<n;i++) {
		for (j=0;j<n;j++)
			printf("%0.0f ",A[i][j]);
		printf("\n");
	}
}

// Free the dynamic memory allocated to the 2D-matrix
void free2dmatrix(double ** M, long n) {
	long i;
	if (!M) return;
	for(i=0;i<n;i++)
		free(M[i]);
	free(M);
}

// Sequential Algorithm
void decomposeSerial(double **A, long n) {
	printf("DECOMPOSE SEQUENTIAL CALLED\n");
	long i,j,k;
	for(k=0;k<n;k++) {
		for(j=k+1;j<n;j++)
			A[k][j]=A[k][j]/A[k][k];
		for(i=k+1;i<n;i++)
			for(j=k+1;j<n;j++)
				A[i][j]=A[i][j] - A[i][k] * A[k][j];
	}
}

// Parallel Algorithm
void decomposeOpenMP(double **A, long n) {
	printf("\nDECOMPOSE PARALLEL CALLED\n");
	long i,j,k,rows,mymin,mymax;
	int pid=0;
	int nprocs;
	#pragma omp parallel shared(A,n,nprocs) private(i,j,k,pid,rows,mymin,mymax)
	{
		#ifdef _OPENMP
			nprocs=omp_get_num_threads();
		#endif
		#ifdef _OPENMP
			pid=omp_get_thread_num();
		#endif
		rows=n/nprocs;
		mymin=pid * rows;
		mymax=mymin + rows - 1;

		if(pid==nprocs-1 && (n-(mymax+1))>0)
			mymax=n-1;

		for(k=0;k<n;k++) {
			if(k>=mymin && k<=mymax){
				for(j=k+1;j<n;j++){
					A[k][j] = A[k][j]/A[k][k];
				}
			}
			#pragma omp barrier
			for(i=(((k+1) > mymin) ? (k+1) : mymin);i<=mymax;i++){
				for(j=k+1;j<n;j++){
					A[i][j] = A[i][j] - A[i][k] * A[k][j];
				}
			}
		}
	}
}

// Initialize  matrix to some random values
void initialize(double **A,long n){
	long i,j, k;
	for(i=0;i<n;i++){
		for(j=i;j<n;j++){
			if(i==j){
				k=i+1;
				A[i][j]=4*k-3;
			}
			else{
				A[i][j]=A[i][i]+1;
				A[j][i]=A[i][i]+1;
			}
		}
	}
}

// Function that calls to create and initialize matrix
double **getMatrix(long size) {
	double **m=make2dmatrix(size);
	initialize(m,size);
	return m;
}

// Parallel Multiply to check if LU=A
void parallelMultiply(double** matrixA, double** matrixB, double** matrixC, long n){
	#pragma omp parallel for
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			for(int k=0; k<n; k++){
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
}

// Breaks the matrix into L and U
void factorize(double **A,long n) {
	long i,j;
	double **u=make2dmatrix(n);
	double **l=make2dmatrix(n);
	for(i=0;i<n;i++) {
		for(j=i;j<n;j++) {
			u[i][j]=A[i][j];
			if(i==j) l[i][j]=1;
			else l[i][j]=0;
		}
	}
	for(i=1;i<n;i++) {
		for(j=0;j<i;j++) {
			l[i][j]=A[i][j];
			u[i][j]=0;
		}
	}
	printf("\nL Matrix :\n\n");
	printmatrix(l,n);
	printf("\nU Matrix :\n\n");
	printmatrix(u,n);
	double **a1=make2dmatrix(n);
	parallelMultiply(l,u,a1,n);
	printf("\nA Matrix after Matrix Mutliplication :\n\n");
	printmatrix(a1,n);
}

int main() {
	long matrix_size;
	printf("Enter the size of matrix (N x N) where N = ");
	scanf("%lu",&matrix_size);
	clock_t begin, end;
	double time_spent;

//Sequential
	double **matrix=getMatrix(matrix_size);
	//printmatrix(matrix,matrix_size);
	begin = clock();
	decomposeSerial(matrix,matrix_size);
	end = clock();
	time_spent = ((double)(end - begin)) / CLOCKS_PER_SEC;
	//factorize(matrix,matrix_size);
	printf("\nDECOMPOSE TIME TAKEN IN SEQUENTIAL : %f seconds\n",time_spent);
	free2dmatrix(matrix,matrix_size);

// Parallel
	long num_threads;
	printf("Enter number of threads ",&num_threads);
	scanf("%ld",&num_threads);
	omp_set_num_threads(num_threads);
	matrix=getMatrix(matrix_size);
	printf("\nInitial Matrix A\n\n");
	printmatrix(matrix,matrix_size);
	begin = clock();
	decomposeOpenMP(matrix,matrix_size);
	end = clock();
	time_spent = ((double)(end - begin)) / CLOCKS_PER_SEC;
	factorize(matrix,matrix_size);
	printf("\nDECOMPOSE TIME TAKEN IN PARALLEL : %f seconds\n",time_spent);
	free2dmatrix(matrix,matrix_size);
	int n;
	scanf("%d",n);
	return 0;
}
