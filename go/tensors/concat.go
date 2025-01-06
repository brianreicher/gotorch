package tensors

import (
	"errors"
	"fmt"
)

// Cat concatenates a slice of tensors along the specified dimension.
// All tensors must have the same shape except in the concatenating dimension.
func Cat(tensors []*Tensor, dim int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("no tensors provided")
	}

	baseShape := tensors[0].Shape
	baseDtype := tensors[0].Dtype
	for _, t := range tensors {
		if len(t.Shape) != len(baseShape) {
			return nil, errors.New("tensors must have the same number of dimensions")
		}
		if t.Dtype != baseDtype {
			return nil, errors.New("tensors must have the same data type")
		}
		for i, dimSize := range t.Shape {
			if i != dim && dimSize != baseShape[i] {
				return nil, fmt.Errorf("dimension mismatch at dim %d", i)
			}
		}
	}

	newShape := make([]int, len(baseShape))
	copy(newShape, baseShape)
	for _, t := range tensors {
		newShape[dim] += t.Shape[dim]
	}

	switch baseDtype {
	case Float32{}:
		concatenatedData, err := concatenateFloat32(tensors, dim)
		if err != nil {
			return nil, err
		}
		return &Tensor{
			Shape:        newShape,
			Data:         concatenatedData,
			Dtype:        Float32{},
			RequiresGrad: tensors[0].RequiresGrad,
			PinMemory:    tensors[0].PinMemory,
		}, nil
	case Float64{}:
		concatenatedData, err := concatenateFloat64(tensors, dim)
		if err != nil {
			return nil, err
		}
		return &Tensor{
			Shape:        newShape,
			Data:         concatenatedData,
			Dtype:        Float64{},
			RequiresGrad: tensors[0].RequiresGrad,
			PinMemory:    tensors[0].PinMemory,
		}, nil
	default:
		return nil, errors.New("unsupported data type")
	}
}

func concatenateFloat32(tensors []*Tensor, dim int) ([]float32, error) {
	var result []float32
	for _, t := range tensors {
		data, err := t.GetData()
		if err != nil {
			return nil, err
		}
		result = append(result, data.([]float32)...)
	}
	return result, nil
}

// Helper to concatenate float64 tensors along a dimension
func concatenateFloat64(tensors []*Tensor, dim int) ([]float64, error) {
	var result []float64
	for _, t := range tensors {
		data, err := t.GetData()
		if err != nil {
			return nil, err
		}
		result = append(result, data.([]float64)...)
	}
	return result, nil
}
