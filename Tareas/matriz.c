#include<stdio.h>
#include<stdlib.h>
#include <time.h>

int main(void){
    int matrix[500][500];
    int i, j, sum;
    
    //Inicialiazción de la matriz con numeros aleatorios en el intervalo [0,9] 
    srand(time(NULL));
    for(i=0; i<500; i++){
        for(j=0; j<500; j++){
            matrix[i][j] = rand() % 10;
        }
    }

    // Forma 1. Acceso a elementos de la matriz de forma convencional
    sum = 0;
    clock_t start1 = clock();
    for(i=0; i<500; i++){
        for(j=0;j<500;j++){
            sum += matrix[i][j];
        }
    }
    printf("Forma 1. Resultado = %d. Tiempo transcurrido: %f[s]\n",sum, (((double)clock() - start1)/CLOCKS_PER_SEC));
    
    //Forma 2. Acceso a elementos de la matriz de la forma *(mat[i]+j)
    sum = 0;
    clock_t start2 = clock();
    for(i=0; i<500;i++){
        for(j=0;j<500;j++){
            sum += *(matrix[i]+j);
        }
    }
    printf("Forma 2. Resultado = %d. Tiempo transcurrido: %f[s]\n", sum, (((double)clock() - start2)/CLOCKS_PER_SEC));

    //Forma 3. Acceso a elementos de la matriz de la forma *(p + n*i+j)
    sum = 0;
    clock_t start3 = clock();

}