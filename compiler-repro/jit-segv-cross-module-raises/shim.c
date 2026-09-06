#include <stdint.h>
#include <stdlib.h>
int32_t ms_stack_alloc(int64_t bytes, void** out_base, void** out_top) {
    void* p = malloc((size_t)bytes + 64);
    if (!p) return 1;
    *out_base = p;
    *out_top = (char*)p + bytes;
    return 0;
}
void ms_stack_free(void* base) { free(base); }
