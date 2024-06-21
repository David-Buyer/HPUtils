package main

import (
	_ "github.com/David-Buyer/HPUtils/backgroundcropper"
)

func main() {

	// Esempio di utilizzo

	// Leggp il file immagine in byte array
	/* imgBytes, err := os.ReadFile("C:\\IPZS_PE\\test_1.jpg")
	if err != nil {
		log.Fatalf("failed to read image file: %v", err)
	}

	// Converto il byte array in un puntatore C
	imgPtr := (*C.uint8_t)(C.CBytes(imgBytes))
	defer C.free(unsafe.Pointer(imgPtr))

	var length C.int
	// Chiamo la funzione Go per elaborare l'immagine
	resultPtr := CropFaceFromBackground(imgPtr, C.int(len(imgBytes)), &length)
	resultBytes := C.GoBytes(unsafe.Pointer(resultPtr), length)

	// Scrivo il risultato in un nuovo file immagine
	err = os.WriteFile("cropped_image.bmp", resultBytes, 0644)
	if err != nil {
		log.Fatalf("failed to write cropped image file: %v", err)
	}

	log.Println("Cropped image saved to: cropped_image.bmp") */
}
