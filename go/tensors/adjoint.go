package tensors

import (
	"errors"
)

func Adjoint(t *Tensor) (*Tensor, error) {
	if len(t.Shape) < 2 {
		return nil, errors.New("Adjoint is only valid for tensors with at least 2 dimensions")
	}

	rows := t.Shape[len(t.Shape)-2]
	cols := t.Shape[len(t.Shape)-1]

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[len(t.Shape)-2] = cols
	newShape[len(t.Shape)-1] = rows

	var newData interface{}
	switch data := t.Data.(type) {
	case []float32:
		newData = make([]float32, len(data))
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				newData.([]float32)[j*rows+i] = data[i*cols+j]
			}
		}
	case []float64:
		newData = make([]float64, len(data))
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				newData.([]float64)[j*rows+i] = data[i*cols+j]
			}
		}
	default:
		return nil, errors.New("unsupported data type for Adjoint")
	}

	return &Tensor{
		Shape:        newShape,
		Data:         newData,
		Dtype:        t.Dtype,
		RequiresGrad: t.RequiresGrad,
		PinMemory:    t.PinMemory,
	}, nil
}
