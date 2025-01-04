package tensors

import (
	"errors"
	"fmt"
)

type Data interface{}

type Dtype interface {
	DataType() string
}

type Float32 struct{}

func (f Float32) DataType() string {
	return "float32"
}

type Float64 struct{}

func (i Float64) DataType() string {
	return "float64"
}

type Device interface {
	Device() string
}

type CPU struct{}

func (c CPU) Device() string {
	return "cpu"
}

type Tensor struct {
	Shape        []int
	Data         Data // Data can hold any type depending on the Dtype ([]float32, []float64)
	Dtype        Dtype
	Device       Device
	RequiresGrad bool
	PinMemory    bool
}

func NewTensor(data Data, shape []int, dtype string, requiresGrad, pinMemory bool) (*Tensor, error) {
	expectedSize := 1
	for _, dim := range shape {
		expectedSize *= dim
	}

	switch dtype {
	case "float32":
		dataFloat32, ok := data.([]float32)
		if !ok {
			return nil, errors.New("data is not of type float32")
		}
		if len(dataFloat32) != expectedSize {
			return nil, errors.New("data length does not match shape")
		}
		return &Tensor{
			Shape:        shape,
			Data:         data,
			Dtype:        Float32{},
			RequiresGrad: requiresGrad,
			PinMemory:    pinMemory,
		}, nil
	case "float64":
		dataFloat64, ok := data.([]float64)
		if !ok {
			return nil, errors.New("data is not of type float64")
		}
		if len(dataFloat64) != expectedSize {
			return nil, errors.New("data length does not match shape")
		}
		return &Tensor{
			Shape:        shape,
			Data:         data,
			Dtype:        Float64{},
			RequiresGrad: requiresGrad,
			PinMemory:    pinMemory,
		}, nil
	default:
		return nil, errors.New("unsupported data type")
	}
}

func (t *Tensor) SetData(newData Data) error {
	expectedSize := 1
	for _, dim := range t.Shape {
		expectedSize *= dim
	}

	switch t.Dtype.DataType() {
	case "float32":
		dataFloat32, ok := newData.([]float32)
		if !ok {
			return errors.New("new data is not of type float32")
		}
		if len(dataFloat32) != expectedSize {
			return errors.New("new data length does not match shape")
		}
		t.Data = newData
	case "float64":
		dataFloat64, ok := newData.([]float64)
		if !ok {
			return errors.New("new data is not of type float64")
		}
		if len(dataFloat64) != expectedSize {
			return errors.New("new data length does not match shape")
		}
		t.Data = newData
	default:
		return errors.New("unsupported data type")
	}
	return nil
}

func (t *Tensor) GetData() (Data, error) {
	switch t.Dtype.DataType() {
	case "float32":
		return t.Data.([]float32), nil
	case "float64":
		return t.Data.([]float64), nil
	default:
		return nil, errors.New("unsupported data type")
	}
}

func (t *Tensor) IsTensor() bool {
	return t != nil
}

func (t *Tensor) IsNonZero() (bool, error) {
	data, err := t.GetData()
	if err != nil {
		return false, err
	}

	switch data := data.(type) {
	case []float32:
		for _, v := range data {
			if v != 0 {
				return true, nil
			}
		}
	case []float64:
		for _, v := range data {
			if v != 0 {
				return true, nil
			}
		}
	default:
		return false, fmt.Errorf("unsupported data type")
	}
	return false, nil
}

func (t *Tensor) Numel() (int, error) {
	data, err := t.GetData()
	if err != nil {
		return 0, err
	}

	switch data := data.(type) {
	case []float32:
		return len(data), nil
	case []float64:
		return len(data), nil
	default:
		return 0, fmt.Errorf("unsupported data type")
	}
}

func NewZeroes(shape []int, dtype Dtype, requiresGrad, pinMemory bool) (*Tensor, error) {
	var (
		data interface{}
		size int = 1
	)

	for _, v := range shape {
		size *= v
	}

	switch dtype.DataType() {
	case "float32":
		data = make([]float32, size)
	case "float64":
		data = make([]float64, size)
	default:
		return nil, errors.New("unsupported data type")
	}

	// Create and return the tensor
	return &Tensor{
		Shape:        shape,
		Data:         data,
		Dtype:        dtype,
		RequiresGrad: requiresGrad,
		PinMemory:    pinMemory,
	}, nil
}
