# Literature notes

These are short notes on the work EnduroSense builds on, grouped by the part of the project each one informs.

- **Status:** the starter set of notes for Phase 0.
- **Legend:** entries marked *(to read)* are summarised from general knowledge and must be checked against the paper before anything from them is cited in the report.

## 1. Dataset and drone energy use

- **Rodrigues, T. A., et al. (2021).** In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery. *Scientific Data.* arXiv:2103.13313. Dataset: <https://doi.org/10.1184/R1/12683453>.
  - Our dataset: 209 flights with controlled speed, payload and altitude. The authors describe the sensors, the ROS time synchronisation and the experimental design.
  - The README says readings are about 5 Hz; our profiling measured about 8 Hz. The report should state which one we use.
- **Rodrigues, T. A., et al. (2022).** Drone flight data reveal energy and greenhouse gas emissions savings for very small package delivery. *Patterns.*
  - Same data. It builds a first-principles energy model plus an ML model covering take-off, cruise and landing.
  - It is the closest prior work to our Model B and the obvious comparison point. *(to read: their phase split and the error they report)*
- **Zhang, J., Campbell, J. F., Sweeney, D. C., & Hupman, A. C. (2021).** Energy consumption models for delivery drones: A comparison and assessment. *Transportation Research Part D.*
  - A survey of drone energy models, showing how widely the predictions of different models disagree.
  - It motivates checking our physics baseline against measured data rather than trusting it. *(to read)*

## 2. Rotor power physics (Model B baseline)

- **Leishman, J. G. (2006).** *Principles of Helicopter Aerodynamics* (2nd ed.). Cambridge University Press.
  - Momentum theory. Hover induced power scales roughly with thrust^1.5.
  - In forward flight, induced power falls as airspeed increases, while parasitic power rises roughly with airspeed^3.
  - This explains the U-shaped energy-versus-speed curve seen in our data (Figure 2a of the plan).
  - Our Phase 4 baseline fits a small number of coefficients to this form.

## 3. Battery remaining-energy estimation (Model A)

- **Plett, G. L. (2015).** *Battery Management Systems, Volume I: Battery Modeling.* Artech House.
  - Covers state of charge, open-circuit voltage, internal resistance and coulomb counting.
  - These are the concepts behind our Model A baselines (voltage lookup, energy counting) and features (voltage sag, internal-resistance estimate).
- **NASA (2018).** Remaining Flying Time Prediction Implementing Battery Prognostics Framework for Electric UAVs. NASA Technical Reports Server, 20180004466.
  - Model-based prediction of remaining flight time for electric UAVs, and the closest prior work to the brief's original framing. *(to read: exact authors, method and assumptions about future load)*

## 4. Machine-learning models compared

- **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *KDD.*
- **Breiman, L. (2001).** Random forests. *Machine Learning, 45*, 5–32.
- **Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term memory. *Neural Computation, 9*(8).
- **Cho, K., et al. (2014).** Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP.* This paper introduced the GRU.

## 5. Uncertainty quantification

- **Koenker, R., & Bassett, G. (1978).** Regression quantiles. *Econometrica, 46*(1). The basis of quantile regression and the pinball loss.
- **Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).** Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS.* The method we use for LSTM/GRU uncertainty.
- **Romano, Y., Patterson, E., & Candès, E. (2019).** Conformalized quantile regression. *NeurIPS.* Adjusts quantile-regression intervals so they reach valid coverage.
- **Angelopoulos, A. N., & Bates, S. (2021).** A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511. A practical guide. Its sections on group-dependent data are relevant to our battery-chain calibration.

## Open questions to settle while reading

1. What mission-energy error did Rodrigues et al. (2022) report? This sets a realistic target for Model B, currently proposed as about 5% average error.
2. Does any prior work combine a remaining-energy prediction with a mission-energy prediction into a probability of success? This checks how novel the feasibility layer is.
3. What reserve conventions exist for small multirotor drones (voltage per cell versus percentage)? This informs the reserve-level question for the guide.
