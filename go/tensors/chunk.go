package tensors

import (
	"errors"
)

// ChunkData reshapes a flat 1D data array into the desired shape.
// It returns a nested slice representing the reshaped data.
func ChunkData(data interface{}, dtype Dtype, shape []int) (interface{}, error) {
	totalElements := 1
	for _, dim := range shape {
		totalElements *= dim
	}

	switch dtype {
	case Float32{}:
		if len(data.([]float32)) != totalElements {
			return nil, errors.New("data size does not match the desired shape")
		}

		chunkedData := reshapeRecursive(data.([]float32), Float32{}, shape)
		return chunkedData, nil
	case Float64{}:
		if len(data.([]float64)) != totalElements {
			return nil, errors.New("data size does not match the desired shape")
		}

		chunkedData := reshapeRecursive(data.([]float64), Float64{}, shape)
		return chunkedData, nil
	default:
		return nil, errors.New("cannot reshape data")
	}
}

func reshapeRecursive(data interface{}, dtype Dtype, shape []int) interface{} {
	if len(shape) == 1 {
		switch dtype {
		case Float32{}:
			return data.([]float32)[:shape[0]]
		case Float64{}:
			return data.([]float64)[:shape[0]]
		default:
			return nil
		}
	}

	chunks := shape[0]
	switch dtype {
	case Float32{}:
		data := data.([]float32)
		chunkSize := len(data) / chunks
		var result []interface{}
		for i := 0; i < chunks; i++ {
			chunk := reshapeRecursive(data[i*chunkSize:(i+1)*chunkSize], Float32{}, shape[1:])
			result = append(result, chunk)
		}

		return result
	case Float64{}:
		data := data.([]float64)
		chunkSize := len(data) / chunks
		var result []interface{}
		for i := 0; i < chunks; i++ {
			chunk := reshapeRecursive(data[i*chunkSize:(i+1)*chunkSize], Float64{}, shape[1:])
			result = append(result, chunk)
		}

		return result
	default:
		return nil
	}
}
