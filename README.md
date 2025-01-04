# GoTorch

🔥**GoTorch**🔥 is a high-performance, CPU-optimized implementation of PyTorch built using Golang at its core, and wrapped for Python use. Designed for highly-concurrent matrix operations and training deep learning models on CPUs, GoTorch leverages Go's concurrency model to achieve efficient computation.

---

## **Outline**

- **High-Performance CPU Training/Inference**: Optimize for parallel computation of matrix operations using Go's concurrency 
- **Ease of Use**: Provide a Python API that mimics PyTorch for seamless integration with existing ML workflows 
- **Extensibility**: Allow future GPU support and integration with other ML tools 

---

## **Roadmap**

### **Phase 1: Foundation**
- [ ] Implement the core `Tensor` data structure 
  - [ ] Basic operations: addition, subtraction, matrix multiplication 
  - [ ] Broadcasting support 
  - [ ] Python wrapping using cgo or PyBind11 

### **Phase 2: Autograd Engine**
- [ ] Autograd engine for tracking and backpropagating gradients 
- [ ] Support gradient computation for basic operations 

### **Phase 3: Neural Network Layers**
- [ ] Implement common layers: Linear, Conv2D, etc 
- [ ] Utility functions for model creation 
- [ ] Loss functions: MSE, Cross-Entropy, etc 

### **Phase 4: Optimization and Training**
- [ ] Optimizers like SGD, Adam, RMSProp 
- [ ] Design a training loop API 
- [ ] Benchmark training speed against PyTorch on CPUs 

### **Phase 5: Extensions and Refinement**
- [ ] Optimize memory usage and concurrency further 
- [ ] Support sparse tensors 
- [ ] Explore GPU integration using CUDA or Vulkan 

---

##  **Development Setup**

### **Requirements**
- Go 1 20+
- Python 3 8+
- PyBind11 or cgo for wrapping Golang in Python

### **Building the Project**
1  Clone the repository:
   ```bash
   git clone https://github com/brianreicher/gotorch git
   cd gotorch
   ```
2  Build the Golang library:
   ```bash
   go build -o libgotorch.so ./
   ```
3  Install the Python package:
   ```bash
   pip install gotorch
   ```

---


## **Performance Goals**
- Achieve faster training on CPUs compared to PyTorch for matrix-heavy workloads 
- Minimize memory overhead using Go's efficient memory management 
- Provide scalability for large models through parallel computation 

---

## **License**
GoTorch is licensed under the MIT License  See the [LICENSE](LICENSE) file for details 

---

## **Acknowledgments**
Inspired by PyTorch, NumPy, and the Go programming language.  Special thanks to the open-source community for their foundational work.

---

## **Contact**
For questions or suggestions, feel free to reach out via GitHub Issues or email at [bk.reicher@gmail.com](bk.reicher@gmail.com)
