# Java Neural Networks from scratch

This project aims to implement Neural Networks algorithms from scratch, as cleanly and modulable as possible.

Library overhead

Here's the call tree of the project, when training a basic MNIST mlps resolver.
As expected, the matrix multiplication takes ~95% of CPU time. 
![image](https://github.com/user-attachments/assets/f9b8339b-b829-4b84-aec5-349e66da833f)
Note that the application only ran for a few minutes, JIT by the JVM might change results while training larger models.
