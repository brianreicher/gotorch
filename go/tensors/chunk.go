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
		return chunkRecursive(data.([]float32), shape), nil
	case Float64{}:
		if len(data.([]float64)) != totalElements {
			return nil, errors.New("data size does not match the desired shape")
		}
		return chunkRecursive(data.([]float64), shape), nil
	default:
		return nil, errors.New("unsupported data type")
	}
}

func chunkRecursive[T any](data []T, shape []int) interface{} {
	// Non recursive case: vector data
	if len(shape) == 1 {
		return data[:shape[0]]
	}

	// Recursive case: for multidimensional tensors (2D, 3D, 4D)
	chunks := shape[0]
	chunkSize := len(data) / chunks
	result := make([]interface{}, chunks)

	for i := 0; i < chunks; i++ {
		result[i] = chunkRecursive(data[i*chunkSize:(i+1)*chunkSize], shape[1:])
	}

	return result
}
