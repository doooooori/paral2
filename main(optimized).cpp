#include <iostream>
#include <math.h>

int main(int argc, char *argv[]){

    double acc = atof(argv[1]);
    int size = strtol(argv[2], NULL, 10);
    int iter_max = strtol(argv[3], NULL, 10);

    double** A = (double**)malloc((size + 2) * sizeof(double*));
    for (int i = 0; i < size + 2; ++i) {
        A[i] = (double*)calloc(size + 2, sizeof(double));
    }
    A[0][0] = 10;
    A[size+1][0] = 20;
    A[0][size+1] = 20;
    A[size+1][size+1] = 30;

    double** Anew = (double**)malloc((size + 2) * sizeof(double*));
    for (int i = 0; i < size + 2; ++i) {
        Anew[i] = (double*)calloc(size + 2, sizeof(double));
    }
    Anew[0][0] = 10;
    Anew[size+1][0] = 20;
    Anew[0][size+1] = 20;
    Anew[size+1][size+1] = 30;

    double step = 10.0/(size+1);
    #pragma acc loop independent
    {
    for (int i = 1; i < size + 1; i++){
        A[i][0] = A[0][0] + step*i;
        A[0][i] = A[0][0] + step*i;
        A[size+1][i] = A[size+1][0] + step*i;
        A[i][size+1] = A[0][size+1] + step*i;
        Anew[i][0] = Anew[0][0] + step*i;
        Anew[0][i] = Anew[0][0] + step*i;
        Anew[size+1][i] = Anew[size+1][0] + step*i;
        Anew[i][size+1] = Anew[0][size+1] + step*i;
    }
    }

    double err = 1;
    int iter = 0;
    double** tmp;
    #pragma acc data copyin(A[:size+2][:size+2],Anew[:size+2][:size+2]) create (err)
    {
    while (err > acc && iter < iter_max){
        iter++;

        if (iter%100 == 0 || iter == 1){
            #pragma acc data present(A,Anew) async
            {
            #pragma acc kernels present(err,A,Anew) async
            {
            err = 0;    
            #pragma acc loop independent collapse(2) reduction(max:err)
            for (int j = 1; j < size + 1; j++)
                for (int i = 1; i < size + 1; i++){
                    Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
                    err = fmax(err, Anew[i][j] - A[i][j]);
                }  
            }
            } 
        } 
        else {
            #pragma acc data present(A, Anew) async
            {
            #pragma acc parallel loop present(A,Anew) independent collapse(2) async
            for (int j = 1; j < size + 1; j++)
                for (int i = 1; i < size + 1; i++)
                    Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
            }
        }
        
         tmp = A;
         A = Anew;
         Anew = tmp;
    
        if (iter%100 == 0 || iter==1){
            #pragma acc wait
            #pragma acc update self(err)
            std::cout << iter << " " << err << std::endl;
        }
    }
    }
    std::cout << iter << " " << err << std::endl;
    return 0;
}
