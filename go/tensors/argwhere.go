package tensors

import "errors"

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
		case 2:
			chunkData := chunkData.([][]float32)
		case 3:
			chunkData := chunkData.([][][]float32)
		case 4:
			chunkData := chunkData.([][][][]float32)
		default:
			return nil, errors.New("tensor dimsion not supported")
		}
	case Float64{}:
		switch dims := len(t.Shape); dims {
		case 1:
			chunkData := chunkData.([]float64)
		case 2:
			chunkData := chunkData.([][]float64)
		case 3:
			chunkData := chunkData.([][][]float64)
		case 4:
			chunkData := chunkData.([][][][]float64)
		default:
			return nil, errors.New("tensor dimsion not supported")
		}
	default:
		return nil, errors.New("datatype not supported")
	}

}
