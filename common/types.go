package common

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int X;
    int Y;
    int Width;
    int Height;
} Rectangle;

typedef void (*ProcessingCallback)(int);
static inline void callProcessingCallback(ProcessingCallback callback, int progress) {
    callback(progress);
}
*/
import "C"
import "unsafe"

// Tipi comuni per i dati C
type (
    Uint8  = C.uint8_t
    Int    = C.int
    Float  = C.float
    Rect   = C.Rectangle
)

// Funzioni di supporto per gestire la memoria C
func Free(ptr unsafe.Pointer) {
    C.free(ptr)
}

func CBytes(b []byte) *Uint8 {
    return (*Uint8)(C.CBytes(b))
}

func GoBytes(ptr unsafe.Pointer, length int) []byte {
    return C.GoBytes(ptr, C.int(length))
}
