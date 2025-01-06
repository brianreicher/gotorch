package tensors

import (
	"errors"
	"fmt"
)

// Reshape reshapes a tensor to the specified shape.
// A single dimension in the shape can be -1, which will be inferred.
func Reshape(t *Tensor, newShape []int) (*Tensor, error) {
	totalElements := 1
	for _, dim := range t.Shape {
		totalElements *= dim
	}

	inferredDim := -1
	newTotalElements := 1
	for i, dim := range newShape {
		if dim == -1 {
			if inferredDim != -1 {
				return nil, errors.New("only one dimension can be -1")
			}
			inferredDim = i
		} else if dim <= 0 {
			return nil, fmt.Errorf("invalid dimension size: %d", dim)
		} else {
			newTotalElements *= dim
		}
	}

	if inferredDim != -1 {
		if totalElements%newTotalElements != 0 {
			return nil, errors.New("cannot infer dimension: inconsistent element count")
		}
		newShape[inferredDim] = totalElements / newTotalElements
		newTotalElements *= newShape[inferredDim]
	}

	if newTotalElements != totalElements {
		return nil, errors.New("total number of elements must remain constant")
	}

	return &Tensor{
		Shape:        newShape,
		Data:         t.Data,
		Dtype:        t.Dtype,
		RequiresGrad: t.RequiresGrad,
		PinMemory:    t.PinMemory,
	}, nil
}
