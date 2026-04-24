## Project Name
**SGWB Parameter Inference in the LISA Band using Simulation-Based Inference**

---

## Overview

This project implements parameter inference for the stochastic gravitational wave background (SGWB) in the LISA frequency band, based on the framework presented in *arXiv:2309.07954*.

Using the **SAQQARA** pipeline and **Simulation-Based Inference (SBI)**, we aim to recover physical parameters of a gravitational wave signal from noisy observational data.

---

## Objectives

The main goals of this project are:

1. Successfully run the `template-powerlaw` branch of SAQQARA
2. Train a neural network to learn the mapping:

   ```
   data → parameters
   ```
3. Perform inference on simulated observations
4. Test the model on a new injection with:

   * amplitude = -12
   * tilt = 0
5. Evaluate the model’s ability to recover true parameters

---

## Methodology

We use **Simulation-Based Inference (SBI)**, which avoids explicit likelihood construction. The workflow is:

1. Sample parameters from prior distributions
2. Generate simulated data using a forward model (simulator)
3. Train a neural network to estimate likelihood ratios
4. Use the trained network to approximate posterior distributions

---

## Experimental Setup

Three sets of experiments are performed:

---

### 1. Data Visualization (Pre-training)

* Plot SGWB signal, noise, and observed data in frequency space
* Validate that the simulator behaves as expected

Purpose:
Understand the structure of the data and signal-to-noise relationship

---

### 2. Inference on Original Injection (In-distribution)

Injection parameters:

```
amplitude = -11
tilt = 0
```

* Train the model on simulated data
* Perform inference on the same type of data

Purpose:
Verify that the model can recover parameters it has effectively learned

---

### 3. Inference on New Injection (Out-of-distribution)

Injection parameters:

```
amplitude = -12
tilt = 0
```

* Generate a new observation with modified amplitude
* Perform inference using the trained model

Purpose:
Evaluate generalization capability of the model

---

## Results

### Spectral Analysis

The observed data is composed of:

```
Data ≈ SGWB + Noise
```

* At low frequencies, the SGWB signal contributes significantly
* At high frequencies, noise dominates

---

### Posterior Distributions

* The recovered posterior distributions for **amplitude** and **tilt** are centered around the true injected values
* Noise parameters (TM, OMS) are less tightly constrained

This demonstrates that the model successfully extracts signal information from noisy data

---

## Environment Setup

Recommended environment:

```bash
python = 3.10
torch = 1.13.1
pytorch_lightning = 1.8.6
torchmetrics = 0.11.0
scipy = 1.10.1
```

Install key dependencies:

```bash
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install pytorch_lightning==1.8.6 torchmetrics==0.11.0
pip install scipy==1.10.1
pip install ipykernel
```

---

## Usage

### Generate Observation

```python
%run generate_observation.py {path_to_config}
```

---

### Train the Model

```python
%run tmnre.py {path_to_config}
```

---

### Perform Inference

```python
new_logratios = trainer.infer(
    network,
    new_obs,
    prior_samples.get_dataloader(batch_size=2048)
)
```

---

## Notes

* On Windows systems, parallel execution may cause instability due to file I/O conflicts. It is recommended to use:

  ```
  run_parallel = False
  njobs = 1
  ```

* Before retraining, remove previous outputs:

  ```
  sgwb_powerlaw/
  ```

---

## Conclusion

This project demonstrates that Simulation-Based Inference can successfully recover SGWB parameters from noisy observations. The trained model accurately reconstructs both the amplitude and spectral tilt, and shows reasonable generalization to unseen parameter configurations.

---

## Summary

The core achievement of this project is:

Recovering hidden physical parameters of a stochastic gravitational wave signal from noisy data using neural-network-based inference.
