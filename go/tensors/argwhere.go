package tensors

import (
	"errors"
	"reflect"
)

func Argwhere(t *Tensor) (*Tensor, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}
	chunkData, err := ChunkData(data, t.Dtype, t.Shape)
	if err != nil {
		return nil, err
	}
	switch t.Dtype {
	case Float32{}:
		switch dims := len(t.Shape); dims {
		case 1:
			chunkData := chunkData.([]float32)
			indexes := make([]int, 0)
			for i, v := range chunkData {
				if v != 0 {
					indexes = append(indexes, i)
				}
			}
			return &Tensor{Shape: []int{len(indexes)},
				Data:         chunkData,
				Dtype:        Float32{},
				RequiresGrad: t.RequiresGrad,
				PinMemory:    t.PinMemory}, nil
		case 2:
			chunkData := chunkData.([][]float32)
			indexes := make([]int, 0)
			for i, row := range chunkData {
				for j, val := range row {
					if val != 0 {
						indexes = append(indexes, i)
						indexes = append(indexes, j)
					}
				}
			}
			return &Tensor{
				Shape:        []int{len(indexes), 2},
				Data:         indexes,
				Dtype:        Float32{},
				RequiresGrad: t.RequiresGrad,
				PinMemory:    t.PinMemory}, nil
		case 3:
			chunkData := chunkData.([][][]float32)
			indexes := make([]int, 0)
			for i, matrix := range chunkData {
				for j, row := range matrix {
					for k, val := range row {
						if val != 0 {
							indexes = append(indexes, i)
							indexes = append(indexes, j)
							indexes = append(indexes, k)
						}
					}
				}
			}
			return &Tensor{
				Shape:        []int{len(indexes), 3},
				Data:         indexes,
				Dtype:        Float32{},
				RequiresGrad: t.RequiresGrad,
				PinMemory:    t.PinMemory,
			}, nil
		case 4:
			chunkData := chunkData.([][][][]float32)
			indexes := make([]int, 0)
			for i, cube := range chunkData {
				for j, matrix := range cube {
					for k, row := range matrix {
						for l, val := range row {
							if val != 0 {
								indexes = append(indexes, i)
								indexes = append(indexes, j)
								indexes = append(indexes, k)
								indexes = append(indexes, l)
							}
						}
					}
				}
			}
			return &Tensor{
				Shape:        []int{len(indexes), 4},
				Data:         indexes,
				Dtype:        Float32{},
				RequiresGrad: t.RequiresGrad,
				PinMemory:    t.PinMemory,
			}, nil
		default:
			return nil, errors.New("tensor dimension not supported")
		}
	case Float64{}:
		switch dims := len(t.Shape); dims {
		case 1:
			chunkData := chunkData.([]float64)
			indexes := make([]int, 0)
			for i, v := range chunkData {
				if v != 0 {
					indexes = append(indexes, i)
				}
			}
			return &Tensor{Shape: []int{len(indexes)},
				Data:         chunkData,
				Dtype:        Float64{},
				RequiresGrad: t.RequiresGrad,
				PinMemory:    t.PinMemory}, nil
		case 2:
			chunkData := chunkData.([][]float64)
			indexes := make([]int, 0)
			for i, row := range chunkData {
				for j, val := range row {
					if val != 0 {
						indexes = append(indexes, i)
						indexes = append(indexes, j)
					}
				}
			}
			return &Tensor{
				Shape:        []int{len(indexes), 2},
				Data:         indexes,
				Dtype:        Float64{},
				RequiresGrad: t.RequiresGrad,
				PinMemory:    t.PinMemory}, nil
		case 3:
			chunkData := chunkData.([][][]float64)
			indexes := make([]int, 0)
			for i, matrix := range chunkData {
				for j, row := range matrix {
					for k, val := range row {
						if val != 0 {
							indexes = append(indexes, i)
							indexes = append(indexes, j)
							indexes = append(indexes, k)
						}
					}
				}
			}

			return &Tensor{
				Shape:        []int{len(indexes), 3},
				Data:         indexes,
				Dtype:        Float64{},
				RequiresGrad: t.RequiresGrad,
				PinMemory:    t.PinMemory,
			}, nil
		case 4:
			chunkData := chunkData.([][][][]float64)
			indexes := make([]int, 0)
			for i, cube := range chunkData {
				for j, matrix := range cube {
					for k, row := range matrix {
						for l, val := range row {
							if val != 0 {
								indexes = append(indexes, i)
								indexes = append(indexes, j)
								indexes = append(indexes, k)
								indexes = append(indexes, l)
							}
						}
					}
				}
			}
			return &Tensor{
				Shape:        []int{len(indexes), 4},
				Data:         indexes,
				Dtype:        Float64{},
				RequiresGrad: t.RequiresGrad,
				PinMemory:    t.PinMemory,
			}, nil
		default:
			return nil, errors.New("tensor dimsion not supported")
		}
	default:
		return nil, errors.New("datatype not supported")
	}

}

func getNonzero[T float32 | float64](arr interface{}) ([]T, error) {
	val := reflect.ValueOf(arr)

	if val.Kind() != reflect.Slice {
		return nil, errors.New("input is not a slice or array")
	}

	var result []T

	var iterate func(reflect.Value)
	iterate = func(v reflect.Value) {
		if v.Kind() != reflect.Slice {
			return
		}
		if v.Len() > 0 && v.Index(0).Kind() != reflect.Slice {
			for i := 0; i < v.Len(); i++ {
				value := v.Index(i).Interface().(T)
				if value != 0 {
					result = append(result, value)
				}
			}
		} else {
			for i := 0; i < v.Len(); i++ {
				iterate(v.Index(i))
			}
		}
	}

	iterate(val)

	return result, nil
}
