package main

/*
#include <stdint.h>
#include <stdlib.h>

*/
//import "C"

import (
	
	_ "log"
	_ "os"
	_ "unsafe"

	_ "github.com/David-Buyer/HPUtils/backgroundcropper"
	_ "github.com/David-Buyer/HPUtils/common"
)

func main() {

	// Esempio di utilizzo

	// Leggo il file immagine in byte array
	/*imgBytes, err := os.ReadFile("C:\\IPZS_PE\\test_1.jpg")
	if err != nil {
		log.Fatalf("failed to read image file: %v", err)
	}

	// Converto il byte array in un puntatore C
	imgPtr := common.CBytes(imgBytes)
    defer common.Free(unsafe.Pointer(imgPtr))

    var length common.Int
    var threshold common.Float = 100 // Imposta un valore di soglia
    var center [2]common.Int

    // Chiamo la funzione Go per elaborare l'immagine
    resultPtr := backgroundcropper.CropFaceFromBackgroundWrapper(imgPtr, common.Int(len(imgBytes)), &length, threshold, &center[0])
    resultBytes := common.GoBytes(unsafe.Pointer(resultPtr), int(length))

	// Scrivo il risultato in un nuovo file immagine
	err = os.WriteFile("C:\\IPZS_PE\\cropped_image.bmp", resultBytes, 0644)
	if err != nil {
		log.Fatalf("failed to write cropped image file: %v", err)
	}

	log.Println("Cropped image saved to: cropped_image.bmp")*/
}
