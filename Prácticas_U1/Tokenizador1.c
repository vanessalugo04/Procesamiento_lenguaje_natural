#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_TOKENS 100
#define MAX_LEN 100

// funcion de tokenizador
void tokenizador1(char texto[]) {
    // medición de tiempo
    clock_t t_ini, t_fin;
    t_ini = clock();

    char tokens[MAX_TOKENS][MAX_LEN]; // matriz para guardar tokens

    int token_index = 0;
    int char_index = 0;

        char token[MAX_LEN];  // construir el token actual
    int j = 0;

    int len = strlen(texto); 

    // recorrer el texto carácter por carácter
    for (int i = 0; i < len; i++) {

        if (texto[i] == ' ' || texto[i] == '.') {
            if (j > 0) {  // evitar tokens vacíos
                token[j] = '\0';  // terminar cadena
                strcpy(tokens[token_index], token); // copia a la matriz
                token_index++;
                j = 0;  // reiniciar token
            }
        } else {
            token[j++] = texto[i];
        }
    }

    // agregar último token si no termina en espacio o punto
    if (j > 0) {
        token[j] = '\0';
        strcpy(tokens[token_index], token);
        token_index++;
    }

    ////////
    t_fin = clock();
    double tiempo = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
    ///////

    printf("Tiempo de ejecución: %f segundos\n", tiempo);
    printf("Tokens encontrados:\n");

    for (int i = 0; i < token_index; i++) {
        printf("%s\n", tokens[i]);
    }
}

int main() {

    char texto[] = "Los alumnos de PLN tienen poca imaginación. Por eso reprobarán";

    tokenizador1(texto);

    return 0;
}