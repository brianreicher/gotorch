package tensors

func Squeeze(t *Tensor) (*Tensor, error) {
	newShape := []int{}
	for _, dim := range t.Shape {
		if dim != 1 {
			newShape = append(newShape, dim)
		}
	}

	return &Tensor{
		Shape:        newShape,
		Data:         t.Data,
		Dtype:        t.Dtype,
		RequiresGrad: t.RequiresGrad,
		PinMemory:    t.PinMemory,
	}, nil
}
