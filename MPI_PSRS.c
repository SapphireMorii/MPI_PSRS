#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

//合并两个已排好序的子数组A[l : m], A[m + 1 : r], 写回A[l : r]
void Merge(int*a,int l,int m,int r)
{
    int n1 = m - l + 1,n2 = r - m;
    int *L = (int*)malloc(sizeof(int)*(n1));
    int *R = (int*)malloc(sizeof(int)*(n2));
    for(int i=0;i<n1;i++)
    {
        L[i] = a[l+i];
    }
    for(int i=0;i<n2;i++)
    {
        R[i] = a[m+1+i];
    }
    int i,j,k;
    i=j=0;
    k=l;
    while(k-l<n1+n2)
    {
        while(L[i]<=R[j]&&i<n1)
        {
            a[k++] = L[i++];
        }
        while(L[i]>R[j]&&j<n2)
        {
            a[k++] = R[j++];
        }
        if(i==n1)
        {
            while(j<n2)
            {
                a[k++] = R[j++];
            }
            break;
        }
        if(j==n2)
        {
            while(i<n1)
            {
                a[k++] = L[i++];
            }
            break;
        }
    }
    free(L);
    free(R);
}

void MergeSort(int *a,int l,int r)
{
    if(l<r)
    {
        int m = (l+r)/2;
        MergeSort(a,l,m);
        MergeSort(a,m+1,r);
        Merge(a,l,m,r);
    }
}

void PSRS(int*a,int n,int id,int num_process)
{
    int *global_samples;
    int *global_sizes;
    int *global_offsets;

    int num_per_process = n / num_process;

    int *samples = (int*)malloc(sizeof(int)*num_process);
    int *pivots = (int*)malloc(sizeof(int)*num_process);
    if(id == 0)
    {
        global_samples = (int*)malloc(sizeof(int)*num_process*num_process);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 均匀划分
    MergeSort(a,id*num_per_process,(id+1)*num_per_process-1);

    //正则采样
    for(int i=0;i<num_process;i++)
    {
        samples[i] = a[id*num_per_process + i*num_per_process/num_process];
    }
    MPI_Gather(samples,num_process,MPI_INT,global_samples,num_process,MPI_INT,0,MPI_COMM_WORLD);

    //采样排序+选择主元
    if(id==0)
    {
        MergeSort(global_samples,0,num_process*num_process-1);
        for(int i=0;i<num_process-1;i++)
        {
            pivots[i] = global_samples[(i+1)*num_process];
        }
        pivots[num_process-1] = INT32_MAX;
    }
    //广播
    MPI_Bcast(pivots,num_process,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int* sizes = (int*)calloc(num_process,sizeof(int));
    int* offsets = (int*)calloc(num_process,sizeof(int));
    int* newsizes = (int*)calloc(num_process,sizeof(int));
    int* newoffsets = (int*)calloc(num_process,sizeof(int));

    //主元划分
    for(int i=0,j=id*num_per_process;j<(id+1)*num_per_process;j++)
    {
        if(a[j]<pivots[i])
        {
            sizes[i]++;
        }else
        {
            sizes[++i]++;
        }
    }
    //全局交换
    MPI_Alltoall(sizes,1,MPI_INT,newsizes,1,MPI_INT,MPI_COMM_WORLD);

    int newdatasize = newsizes[0];
    for(int i=1;i<num_process;i++)
    {
        offsets[i] = offsets[i-1] + sizes[i-1];
        newoffsets[i] = newoffsets[i-1] + newsizes[i-1];
        newdatasize += newsizes[i];
    }

    int* newdata = (int*)malloc(sizeof(int)*newdatasize);

    MPI_Alltoallv(&(a[id*num_per_process]),sizes,offsets,MPI_INT,newdata,newsizes,newoffsets,MPI_INT,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    MergeSort(newdata,0,newdatasize-1);
    MPI_Barrier(MPI_COMM_WORLD);

    if(id==0)
    {
        global_sizes = (int*)calloc(num_process,sizeof(int));
    }
    MPI_Gather(&newdatasize,1,MPI_INT,global_sizes,1,MPI_INT,0,MPI_COMM_WORLD);
    if(id==0)
    {
        global_offsets = (int*)calloc(num_process,sizeof(int));
        for(int i=1;i<num_process;i++)
        {
            global_offsets[i]=global_offsets[i-1]+global_sizes[i-1];
        }
    }
    MPI_Gatherv(newdata,newdatasize,MPI_INT,a,global_sizes,global_offsets,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    free(samples);
    free(pivots);
    free(sizes);
    free(offsets);
    free(newdata);
    free(newsizes);
    free(newoffsets);
    if(id==0)
    {
        free(global_offsets);
        free(global_samples);
        free(global_sizes);
    }
}

int verify(int*A,int*B,int len)
{
    for(int i=0;i<len;i++)
    {
        if(A[i]!=B[i]) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]){
    //int A[27] = {15, 46, 48, 93, 39, 6, 72, 91, 14, 36, 69, 40, 89, 61, 97, 12, 21, 54, 53, 97, 84, 58, 32, 27, 33, 72, 20};
    //int A[81] = {13, 38, 76, 99, 94, 82, 86, 46, 14, 68, 84, 4, 78, 92, 26, 25, 57, 11, 59, 54, 75, 60, 70, 41, 23, 66, 93, 1, 100, 9, 22, 3, 58, 63, 51, 90, 29, 73, 27, 47, 0, 71, 43, 89, 67, 2, 8, 31, 19, 37, 24, 49, 62, 87, 12, 56, 32, 15, 77, 61, 95, 64, 55, 10, 7, 45, 30, 83, 28, 96, 44, 98, 17, 48, 34, 97, 35, 20, 80, 16, 53};
    int length = 10000000;
    int *A = (int*)malloc(sizeof(int)*length);
    int *B = (int*)malloc(sizeof(int)*length);
    srand(time(NULL));
    for(int i=0;i<length;i++)
    {
        A[i] = B[i] = rand();
    }
    double t1, t2;
    int id, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes); //获取进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &id); //获取当前进程id
    if(id == 0)
        t1 = MPI_Wtime();
    PSRS(A, length, id, num_processes);
    if(id == 0){
        t2 = MPI_Wtime();
        printf("MPI time: %lfs\n", t2 - t1);
        // for(int i = 0; i < 27; i++)
        //     printf("%d ", A[i]);
        // printf("\n");
    }
    //MPI_Finalize();
    if(id==0)
    {
        double st,en;
        //st = time(NULL);
        st = MPI_Wtime();
        MergeSort(B,0,length-1);
        en = MPI_Wtime();
        printf("common mergesort time: %lfs\n",en-st);
        //en = time(NULL);
        //printf("common mergesort time: %lfs\n",difftime(en,st));
        // for(int i = 0; i < 27; i++)
        //         printf("%d ", B[i]);
        //printf("\n");
        if(verify(A,B,length))
            printf("Have same result\n");
    }
    MPI_Finalize();
    return 0;
}