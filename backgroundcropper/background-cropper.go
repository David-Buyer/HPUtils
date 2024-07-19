package backgroundcropper

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

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/David-Buyer/HPUtils/common"
	_ "github.com/disintegration/imaging"
	"golang.org/x/image/bmp"
)

var onProcessing func(int)
var percCompletion float32 = 0
var forcedRatio float32 = 1.33
var processingCallback C.ProcessingCallback
var progressQueue chan int
var stopWorker chan struct{}

//export CFree
func CFree(ptr *C.uint8_t) {
	close(stopWorker) // Ferma il worker
	C.free(unsafe.Pointer(ptr))
}

func findMaxDensityCenter(data []byte, windowSize int, step int) [2]int {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		log.Fatalf("failed to decode image: %v", err)
	}
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// Crea e popola l'immagine integrale
	integral := make([][]int, height+1)
	for i := range integral {
		integral[i] = make([]int, width+1)
	}

	for y := 1; y <= height; y++ {
		for x := 1; x <= width; x++ {
			r, g, b, _ := img.At(x-1, y-1).RGBA()
			pixelValue := 0
			if r>>8 < 250 && g>>8 < 250 && b>>8 < 250 {
				pixelValue = 1
			}
			integral[y][x] = pixelValue + integral[y-1][x] + integral[y][x-1] - integral[y-1][x-1]
		}
	}

	maxDensity := -1
	var maxDensityCenter [2]int
	var totalDensity int
	var windowCount int

	// Mappa per memorizzare le densità e le rispettive coordinate
	densityMap := make(map[int][][2]int)

	// Utilizza l'immagine integrale per calcolare la densità in ogni finestra mobile
	for y := 0; y <= height-windowSize; y += step {
		for x := 0; x <= width-windowSize; x += step {
			x1, y1 := x, y
			x2, y2 := x+windowSize, y+windowSize
			density := integral[y2][x2] - integral[y1][x2] - integral[y2][x1] + integral[y1][x1]

			if density > maxDensity {
				totalDensity += density
				windowCount++

				// Aggiungi le coordinate alla mappa della densità
				densityMap[density] = append(densityMap[density], [2]int{x, y})
				maxDensity = density
				maxDensityCenter = [2]int{x, y}
			}
		}
	}

	averageDensity := float64(totalDensity) / float64(windowCount)

	closestDensity := maxDensity
	minDiff := float64(maxDensity)
	for density := range densityMap {
		diff := math.Abs(float64(density) - averageDensity)
		if diff < minDiff {
			minDiff = diff
			closestDensity = density
		}
	}

	// Prendi una delle coordinate associate alla densità più vicina alla media
	averageDensityCenter := densityMap[closestDensity][0]

	log.Printf("Max density center: (%d, %d) with density %d", maxDensityCenter[0], maxDensityCenter[1], maxDensity)
	return averageDensityCenter
}

//export SetProcessingCallback
func SetProcessingCallback(callback C.ProcessingCallback) {
	processingCallback = callback
	log.Println("Callback set")
	progressQueue = make(chan int, 100)
	stopWorker = make(chan struct{})

	// Avvia un worker per processare la coda di progressi
	go func() {
		for {
			select {
			case progress := <-progressQueue:
				if processingCallback != nil {
					C.callProcessingCallback(processingCallback, C.int(progress))
				}
			case <-stopWorker:
				return
			}
		}
	}()
}

func RaiseOnProcessing(progress int) {
	select {
	case progressQueue <- progress:
	default:
	}
}

func CropFaceFromBackgroundWrapper(data *common.Uint8, length common.Int, resultLength *common.Int, bnActivationThreshold common.Float, centerOffset *common.Int, result *common.Int) *C.uint8_t {
	return CropFaceFromBackground((*C.uint8_t)(data), (C.int)(length), (*C.int)(resultLength), (C.float)(bnActivationThreshold), (*C.int)(centerOffset), (*C.int)(result))
}

//export CropFaceFromBackground
func CropFaceFromBackground(data *C.uint8_t, length C.int, resultLength *C.int, bnActivationThreshold C.float, centerOffset *C.int, result *C.int) *C.uint8_t {
	otsuOffset := 30
	centerOffsetArray := (*[2]C.int)(unsafe.Pointer(centerOffset))
	byteArray := C.GoBytes(unsafe.Pointer(data), length)
	img, _, err := image.Decode(bytes.NewReader(byteArray))
	if err != nil {
		log.Fatalf("failed to decode image: %v", err)
	}

	*result = 1

	if IsBNColorRatio(img, float32(bnActivationThreshold), true) {
		*result = 0
		threshold := OtsuThreshold(img)
		binarizedImg := BinarizeImage(img, threshold, otsuOffset)
		faceRectangle := FindLargestBlob(binarizedImg)

		if faceRectangle != (image.Rectangle{}) {
			croppedImg := cropImage(img, faceRectangle)
			buf, err := encodeToBitmap(croppedImg)
			if err != nil {
				log.Fatalf("failed to encode cropped image: %v", err)
			}

			length = C.int(len(buf))
			byteArray = buf
		}
	}

	radius := 100 // Modificabile secondo le necessità
	center := findMaxDensityCenter(byteArray, radius, 20)

	// Imposta i valori di centerOffset
	centerOffsetArray[0] = C.int(center[0])
	centerOffsetArray[1] = C.int(center[1])

	*resultLength = length
	return (*C.uint8_t)(C.CBytes(byteArray))
}

func cropImage(img image.Image, rect image.Rectangle) image.Image {
	cropped := image.NewRGBA(rect)
	draw.Draw(cropped, rect, img, rect.Min, draw.Src)
	return cropped
}

func encodeToBitmap(img image.Image) ([]byte, error) {
	buf := new(bytes.Buffer)
	err := bmp.Encode(buf, img)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func IsBNColorRatio(img image.Image, ratio float32, white bool) bool {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	col := uint8(250)
	if !white {
		col = 0
	}
	whitePixelCount := 0
	totalPixels := width * height

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			r8, g8, b8 := uint8(r>>8), uint8(g>>8), uint8(b>>8)

			if white {
				if r8 >= col && g8 >= col && b8 >= col {
					whitePixelCount++
				}
			} else {
				if r8 == col && g8 == col && b8 == col {
					whitePixelCount++
				}
			}
		}
	}

	whitePercentage := float32(whitePixelCount) / float32(totalPixels) * 100.0
	return whitePercentage > ratio-1
}

func saveImage(img image.Image, filePath string) error {
	outFile, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	err = png.Encode(outFile, img)
	if err != nil {
		return err
	}

	log.Printf("Image saved to %s", filePath)
	return nil
}

func BinarizeImage(img image.Image, threshold int, otsuOffset int) image.Image {
	bounds := img.Bounds()
	output := image.NewNRGBA(bounds)

	var wg sync.WaitGroup

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		wg.Add(1)
		go func(y int) {
			defer wg.Done()
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, g, b, _ := img.At(x, y).RGBA()
				r8, g8, b8 := uint8(r>>8), uint8(g>>8), uint8(b>>8)

				// Calcolo con i coefficienti standard di conversione YUV
				gray := (int(r8)*299 + int(g8)*587 + int(b8)*114) / 1000

				var binColor color.NRGBA
				if gray < threshold+otsuOffset {
					binColor = color.NRGBA{0, 0, 0, 255} // Nero
				} else {
					binColor = color.NRGBA{255, 255, 255, 255} // Bianco
				}
				output.Set(x, y, binColor)
			}
		}(y)
	}

	wg.Wait()

	// Salvo l'immagine binarizzata in un file per debugging
	//saveImage(output, "C:\\IPZS_PE\\binarized_image.png")

	if IsBNColorRatio(output, 100.0, false) {
		return BinarizeImage(img, threshold, max(0, otsuOffset-10))
	}

	return output
}

func OtsuThreshold(img image.Image) int {
	histogram := [256]int{}
	total := 0
	bounds := img.Bounds()

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			grayColor := color.GrayModel.Convert(img.At(x, y)).(color.Gray)
			histogram[grayColor.Y]++
			total++
		}
	}

	sum := 0
	for i := 0; i < 256; i++ {
		sum += i * histogram[i]
	}

	sumB := 0
	wB := 0
	wF := 0
	varMax := 0.0
	threshold := 0

	for i := 0; i < 256; i++ {
		wB += histogram[i]
		if wB == 0 {
			continue
		}
		wF = total - wB
		if wF == 0 {
			break
		}

		sumB += i * histogram[i]
		mB := float64(sumB) / float64(wB)
		mF := float64(sum-sumB) / float64(wF)
		varBetween := float64(wB) * float64(wF) * (mB - mF) * (mB - mF)

		if varBetween > varMax {
			varMax = varBetween
			threshold = i
		}
	}
	return threshold
}

func FindLargestBlob(img image.Image) image.Rectangle {
	searchDistance := 5
	dilateRadius := 10

	start := time.Now()
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	labels := make([][]int, width)
	for i := range labels {
		labels[i] = make([]int, height)
	}
	uf := NewUnionFind(width * height)
	var labelCount int32 = 1

	unitPerc := 87.0 / float32(width)
	preProcessedImage := erodeAndDilate(img, dilateRadius)

	var mutex sync.Mutex
	blockSize := 200 // Numero di colonne per blocco
	numBlocks := (width + blockSize - 1) / blockSize
	jobs := make(chan int, numBlocks)
	results := make(chan bool, numBlocks)
	numWorkers := 8

	// Avvio dei worker
	for w := 0; w < numWorkers; w++ {
		go worker(w, jobs, results, preProcessedImage, labels, uf, &labelCount, unitPerc, blockSize, &mutex, searchDistance)
	}

	// Invio dei blocchi di lavoro
	for block := 0; block < numBlocks; block++ {
		jobs <- block
	}
	close(jobs)

	// Attendi che tutti i worker completino il loro lavoro
	for i := 0; i < numBlocks; i++ {
		<-results
	}

	blobs := make(map[int][]image.Point)
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			label := uf.Find(labels[x][y])
			if label != 0 {
				blobs[label] = append(blobs[label], image.Point{X: x, Y: y})
			}
		}
	}

	log.Printf("Number of blobs before removing sparse lines: %d", len(blobs))
	blobs = RemoveSparseLines(blobs, 30)
	log.Printf("Number of blobs after removing sparse lines: %d", len(blobs))

	unitPerc = 13.0 / float32(len(blobs))

	maxPixelCount := 0
	var largestBlobRect image.Rectangle
	for label, blob := range blobs {
		percCompletion += unitPerc
		RaiseOnProcessing(int(percCompletion))
		log.Printf("Blob %d size: %d", label, len(blob))

		boundingRect := GetBoundingRectangle(blob)
		if len(blob) > maxPixelCount {
			maxPixelCount = len(blob)
			largestBlobRect = boundingRect
		}
	}
	log.Printf("Tempo esecuzione: %v secondi", time.Since(start).Seconds())
	return largestBlobRect
}

func worker(id int, jobs <-chan int, results chan<- bool, img image.Image, labels [][]int, uf *UnionFind, labelCount *int32, unitPerc float32, blockSize int, mutex *sync.Mutex, searchDistance int) {
	_ = id
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	for block := range jobs {
		start := block * blockSize
		end := start + blockSize
		if end > width {
			end = width
		}

		for x := start; x < end; x++ {
			mutex.Lock()
			percCompletion += unitPerc
			RaiseOnProcessing(int(percCompletion))
			mutex.Unlock()

			for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
				gray, _, _, _ := img.At(x, y).RGBA()
				if uint8(gray>>8) == 0 { // Verifica se il valore del pixel grigio è 0
					label := int(atomic.LoadInt32(labelCount))
					neighborLabels := []int{}

					for dy := -searchDistance; dy <= searchDistance; dy++ {
						for dx := -searchDistance; dx <= searchDistance; dx++ {
							nx, ny := x+dx, y+dy
							if nx >= 0 && nx < width && ny >= 0 && ny < height && labels[nx][ny] != 0 {
								neighborLabels = append(neighborLabels, labels[nx][ny])
							}
						}
					}

					// if x > 0 && labels[x-1][y] != 0 {
					//     neighborLabels = append(neighborLabels, labels[x-1][y])
					// }
					// if y > 0 && labels[x][y-1] != 0 {
					//     neighborLabels = append(neighborLabels, labels[x][y-1])
					// }
					// if x < width-1 && labels[x+1][y] != 0 {
					//     neighborLabels = append(neighborLabels, labels[x+1][y])
					// }
					// if y < height-1 && labels[x][y+1] != 0 {
					//     neighborLabels = append(neighborLabels, labels[x][y+1])
					// }

					if len(neighborLabels) > 0 {
						label = neighborLabels[0]
						for _, l := range neighborLabels {
							if l < label {
								label = l
							}
						}
						mutex.Lock()
						for _, l := range neighborLabels {
							uf.Union(label, l)
						}
						mutex.Unlock()
					} else {
						label = int(atomic.AddInt32(labelCount, 1))
					}

					labels[x][y] = label

				}
			}
		}
		results <- true
	}
}

func MergeLabels(neighborLabels []int, smallestLabel int, labels [][]int, blobs map[int][]image.Point) {
	for _, label := range neighborLabels {
		if label != smallestLabel {
			blobs[smallestLabel] = append(blobs[smallestLabel], blobs[label]...)
			delete(blobs, label)
			for x := range labels {
				for y := range labels[x] {
					if labels[x][y] == label {
						labels[x][y] = smallestLabel
					}
				}
			}
		}
	}
}

func erodeAndDilate(img image.Image, radius int) image.Image {
	bounds := img.Bounds()
	//eroded := image.NewGray(bounds)
	dilated := image.NewRGBA(bounds)

	white := color.RGBA{255, 255, 255, 255}
	draw.Draw(dilated, bounds, &image.Uniform{white}, image.Point{}, draw.Src)
	
	//Accedo ai pixel direttamente per incrementare la performance
	src := img.(*image.NRGBA)

	//Erosione
	// for y := bounds.Min.Y + radius; y < bounds.Max.Y-radius; y++ {
	// 	for x := bounds.Min.X + radius; x < bounds.Max.X-radius; x++ {
	// 		min := uint8(250)
	// 		for dy := -radius; dy <= radius; dy++ {
	// 			for dx := -radius; dx <= radius; dx++ {
	// 				// Controlla direttamente se il pixel è nero o bianco
	// 				if grayColor, ok := img.At(x+dx, y+dy).(color.Gray); ok {
	// 					if grayColor.Y >= min {
	// 						min = grayColor.Y
	// 					}
	// 				}
	// 			}
	// 		}
	// 		eroded.SetGray(x, y, color.Gray{Y: min})
	// 	}
	// }

	
	//Dilatazione
	for y := bounds.Min.Y + radius; y < bounds.Max.Y-radius; y++ {
		for x := bounds.Min.X + radius; x < bounds.Max.X-radius; x++ {
			// Verifica se il pixel corrente è nero
			r, g, b, _ := src.NRGBAAt(x, y).RGBA()
			isBlack := r>>8 == 0 && g>>8 == 0 && b>>8 == 0

			if isBlack {
				// Dilata i pixel neri
				for dy := -radius; dy <= radius; dy++ {
					for dx := -radius; dx <= radius; dx++ {
						xx := x + dx
						yy := y + dy
						if xx >= bounds.Min.X && xx < bounds.Max.X && yy >= bounds.Min.Y && yy < bounds.Max.Y {
							dilated.SetRGBA(xx, yy, color.RGBA{0, 0, 0, 255})
						}
					}
				}
			}
		}
	}

	//saveImage(dilated, "C:\\IPZS_PE\\binarized_image_2.png")
	return dilated
}

func GetBoundingRectangle(points []image.Point) image.Rectangle {
	minX, minY := points[0].X, points[0].Y
	maxX, maxY := points[0].X, points[0].Y

	for _, p := range points {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}

	width := maxX - minX + 1
	height := maxY - minY + 1
	ratio := float32(height) / float32(width)

	if ratio < forcedRatio || ratio > (forcedRatio+0.05) {
		newHeight := int(float32(width) * forcedRatio)
		minY -= newHeight - height
		height = newHeight
	}

	return image.Rect(minX, max(0, minY), minX+width, minY+height)
}

func RemoveSparseLines(blobs map[int][]image.Point, threshold int) map[int][]image.Point {
	refinedBlobs := make(map[int][]image.Point)

	for label, points := range blobs {
		rowDensity := make(map[int]int)
		colDensity := make(map[int]int)

		for _, point := range points {
			rowDensity[point.Y]++
			colDensity[point.X]++
		}

		refinedPoints := []image.Point{}
		for _, point := range points {
			if rowDensity[point.Y] > threshold && colDensity[point.X] > threshold {
				refinedPoints = append(refinedPoints, point)
			}
		}

		if len(refinedPoints) > 0 {
			refinedBlobs[label] = refinedPoints
		}
	}

	return refinedBlobs
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
