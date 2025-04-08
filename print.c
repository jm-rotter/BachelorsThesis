
#include <stdlib.h>
#include <stdio.h>

int main() {
    // Allocate some memory
    void *ptr1 = malloc(100);  // This should trigger the override

    // Allocate some more memory
    void *ptr2 = malloc(200);  // This should also trigger the override

    // Free the memory
    free(ptr1);
    free(ptr2);

    return 0;
}
