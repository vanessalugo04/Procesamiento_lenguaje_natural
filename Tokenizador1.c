#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_TOKENS 100
#define MAX_LEN 100

void tokenizador1(char texto[]) {
    clock_t t_ini, t_fin;
    t_ini = clock();

    char tokens[MAX_TOKENS][MAX_LEN];
    int token_index = 0;
    int char_index = 0;

    char token[MAX_LEN];
    int j = 0;

    int len = strlen(texto);

    for (int i = 0; i < len; i++) {

        if (texto[i] == ' ' || texto[i] == '.') {
            if (j > 0) {  
                token[j] = '\0';  
                strcpy(tokens[token_index], token);
                token_index++;
                j = 0;  
            }
        } else {
            token[j++] = texto[i];
        }
    }

    if (j > 0) {
        token[j] = '\0';
        strcpy(tokens[token_index], token);
        token_index++;
    }

    t_fin = clock();
    double tiempo = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;

    printf("Tiempo de ejecucion: %f segundos\n", tiempo);
    printf("Tokens encontrados:\n");

    for (int i = 0; i < token_index; i++) {
        printf("%s\n", tokens[i]);
    }
}

int main() {

    char texto[] = "Los alumnos de PLN tienen poca imaginacion. Por eso reprobaran";

    tokenizador1(texto);

    return 0;
}