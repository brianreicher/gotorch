package tensors

import (
	"errors"
	"fmt"
)

func Heavyside(input, values Tensor) (Tensor, error) {
	if !heavysideValidShape(input, values) {
		return Tensor{}, errors.New("tensors are not of valid size")
	}
	if input.Dtype.DataType() != values.Dtype.DataType() {
		return Tensor{}, errors.New("input and value tensor are not of the same dtype")
	}

	dtype := input.Dtype
	inputData, err := input.GetData()
	if err != nil {
		return Tensor{}, err
	}

	valueData, err := values.GetData()
	if err != nil {
		return Tensor{}, err
	}

	switch dtype {
	case Float32{}:
		inputData := inputData.([]float32)
		valueData := valueData.([]float32)
		outputData := make([]float32, len(inputData))
		copy(outputData, inputData)

		if len(valueData) == 1 {
			for i, v := range outputData {
				if v == 0. {
					outputData[i] = valueData[0]
				}
			}
		} else {
			for i, v := range outputData {
				if v == 0. {
					outputData[i] = valueData[i]
				}
			}
		}
	case Float64{}:
		inputData := inputData.([]float64)
		valueData := valueData.([]float64)
		outputData := make([]float64, len(inputData))
		copy(outputData, inputData)

		if len(valueData) == 1 {
			for i, v := range outputData {
				if v == 0. {
					outputData[i] = valueData[0]
				}
			}
		} else {
			for i, v := range outputData {
				if v == 0. {
					outputData[i] = valueData[i]
				}
			}
		}
	default:
		return Tensor{}, fmt.Errorf("unsupported data type")
	}
	return Tensor{
		Shape:        input.Shape,
		Data:         inputData,
		Dtype:        input.Dtype,
		RequiresGrad: input.RequiresGrad,
		PinMemory:    input.PinMemory,
	}, nil
}

func heavysideValidShape(input, values Tensor) bool {
	if len(values.Shape) == 1 {
		return true
	}

	inputSize, valueSize := 1, 1

	for _, v := range input.Shape {
		inputSize *= v
	}

	for _, v := range values.Shape {
		valueSize *= v
	}
	return inputSize == valueSize
}
