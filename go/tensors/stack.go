package tensors

import "errors"

// Stack stacks a sequence of tensors along a new dimension.
// All input tensors must have the same shape.
func Stack(tensors []*Tensor, dim int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("no tensors provided for stacking")
	}

	baseShape := tensors[0].Shape
	baseDtype := tensors[0].Dtype
	for _, t := range tensors {
		if !equalShapes(baseShape, t.Shape) {
			return nil, errors.New("all tensors must have the same shape")
		}
		if t.Dtype != baseDtype {
			return nil, errors.New("all tensors must have the same dtype")
		}
	}

	newShape := make([]int, len(baseShape)+1)
	copy(newShape, baseShape)
	if dim < 0 {
		dim += len(newShape)
	}
	if dim < 0 || dim > len(newShape) {
		return nil, errors.New("dimension out of range")
	}
	newShape = append(newShape[:dim], append([]int{len(tensors)}, newShape[dim:]...)...)

	stackedData := stackData(tensors, baseDtype, dim)

	return &Tensor{
		Shape:        newShape,
		Data:         stackedData,
		Dtype:        baseDtype,
		RequiresGrad: tensors[0].RequiresGrad,
		PinMemory:    tensors[0].PinMemory,
	}, nil
}

func stackData(tensors []*Tensor, dtype Dtype, dim int) interface{} {
	switch dtype {
	case Float32{}:
		return stackRecursive(tensors, dim, Float32{}).([]interface{})
	case Float64{}:
		return stackRecursive(tensors, dim, Float64{}).([]interface{})
	default:
		return nil
	}
}

func stackRecursive(tensors []*Tensor, dim int, dtype Dtype) interface{} {
	if len(tensors) == 1 {
		return tensors[0].Data
	}

	for _, t := range tensors {
		if len(t.Shape) != len(tensors[0].Shape) {
			panic("all tensors must have the same number of dimensions")
		}
		for i := range t.Shape {
			if i != dim && t.Shape[i] != tensors[0].Shape[i] {
				panic("all tensors must have the same shape except in the stacking dimension")
			}
		}
	}

	result := make([]interface{}, len(tensors))
	for i, t := range tensors {
		data := t.Data
		if len(t.Shape) > 1 && dim > 0 {
			subTensors := extractSubTensors(t, dim)
			result[i] = stackRecursive(subTensors, dim-1, dtype)
		} else {
			result[i] = data
		}
	}

	return result
}
func extractSubTensors(t *Tensor, dim int) []*Tensor {
	subTensors := []*Tensor{}
	shape := t.Shape
	size := shape[dim]
	data := t.Data.([]interface{})

	for i := 0; i < size; i++ {
		subTensors = append(subTensors, &Tensor{
			Shape:        append(shape[:dim], shape[dim+1:]...),
			Data:         data[i],
			Dtype:        t.Dtype,
			RequiresGrad: t.RequiresGrad,
			PinMemory:    t.PinMemory,
		})
	}

	return subTensors
}

func equalShapes(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}
