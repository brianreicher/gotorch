package tensors

import "errors"

func Transpose(t *Tensor, dim1, dim2 int) (*Tensor, error) {
	if dim1 < 0 || dim1 >= len(t.Shape) || dim2 < 0 || dim2 >= len(t.Shape) {
		return nil, errors.New("invalid dimensions for transpose")
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[dim1], newShape[dim2] = newShape[dim2], newShape[dim1]

	transposedData := transposeRecursive(t.Data, t.Shape, dim1, dim2)

	return &Tensor{
		Shape:        newShape,
		Data:         transposedData,
		Dtype:        t.Dtype,
		RequiresGrad: t.RequiresGrad,
		PinMemory:    t.PinMemory,
	}, nil
}

func transposeRecursive(data interface{}, shape []int, dim1, dim2 int) interface{} {
	if len(shape) <= 1 {
		return data
	}

	if len(shape) == 2 {
		rows := shape[dim1]
		cols := shape[dim2]

		result := make([][]float32, cols)
		for i := range result {
			result[i] = make([]float32, rows)
		}

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result[j][i] = data.([][]float32)[i][j]
			}
		}
		return result
	}

	outerDim := shape[0]
	innerShape := shape[1:]
	result := make([]interface{}, outerDim)

	for i := 0; i < outerDim; i++ {
		result[i] = transposeRecursive(data.([]interface{})[i], innerShape, dim1-1, dim2-1)
	}

	return result
}
