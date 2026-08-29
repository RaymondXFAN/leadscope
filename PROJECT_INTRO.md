# LeadScope — Project Introduction (English)

## One-line Summary

LeadScope is a secondary-analysis toolkit and companion paper code for the openly shared dyadic dance motion-capture dataset of Bigand et al. (2024), systematically identifying the coupling structure and leader–follower lag of dancing dyads.

## Short Description

This project focuses on time–frequency analysis of dyadic dance coordination under the constraint of a 5 Hz sampling rate. It comprises a core lagged cross-correlation library (`lagcc`) and twelve analysis scripts, examining the data from two complementary perspectives: the time domain (lead–follower lag, sliding-window time-varying lag, and song symmetry) and the frequency domain (band-wise coherence). Statistically, dyads (35 dancer pairs) are treated as the independent unit, with Wilcoxon paired tests, FDR correction, and IAAFT surrogate null distributions ensuring rigor. Key findings include the coupling difference between same-song vs different-song conditions (Cohen's *d*\_av = 1.128), the enhancement of band coupling under visual contact (fast band *d*\_av = 2.314), and the complementary insight that temporal-lag directionality and frequency-domain coupling are mutually informative under sampling-rate limitations. All code, figures, and reproduction scripts are released with the package.
