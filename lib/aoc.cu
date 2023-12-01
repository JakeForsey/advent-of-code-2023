#include <stdio.h>

#include "aoc.h"

char *read_file(char file_path[]) {
    FILE *file = fopen((char*) file_path, "r");
    fseek(file, 0L, SEEK_END);
    long n = ftell(file);
    fseek(file, 0L, SEEK_SET);
    char *text = (char*)calloc(n, sizeof(char));
    fread(text, sizeof(char), n, file);
    fclose(file);
    return text;
}
