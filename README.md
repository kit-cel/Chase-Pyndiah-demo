# Chase–Pyndiah Demo

This repository accompanies the paper:

> **Improved Chase–Pyndiah Decoding for Product Codes with Scaled Messages**

It provides simulation tools to evaluate and compare decoding algorithms for product codes, including the **classical Chase–Pyndiah decoder** and the **enhanced variant** proposed in the paper.

---

## Abstract

We propose an enhanced Chase–Pyndiah decoder that scales extrinsic messages based on the confidence of the component decoder.  
The proposed method achieves a performance gain of approximately **0.1 dB** over the original algorithm, with **negligible increase in computational complexity**.

---
### `SISOsim.cpp`

C++ simulation framework for product code decoding.

Implements:
- Original Chase–Pyndiah decoder  
- Enhanced Chase–Pyndiah decoder (proposed method)
---

### `CP_parameter_opt.py`

Python script for optimizing the parameters alpha and beta for the Chase–Pyndiah decoder


---



# Funding Acknowledgment
This work has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 101001899).


