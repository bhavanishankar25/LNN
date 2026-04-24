# Future LNN Projects

## Current: Solar Wind → Geomagnetic Storm Prediction (v4)

DSCOVR satellite solar wind data → predict Kp geomagnetic storm index at Earth (1 hour ahead).

**Status:** v3 validated (0.926 correlation on 2025 data). v4 ready for training on GPU cluster.

**v4 specs:** 16 raw OMNI2 features + 17 engineered = 33 total. 96 hidden x 2 layers. Ensemble of 5. 72h sequences. 60 years of training data (1963-2024).

---

## Next: Distillation Project

The long-term goal is to distill large time-series foundation models into the LNN:

- **Teachers:** Chronos-2 (Amazon), TimesFM 2.5 (Google), Moirai-2 (Salesforce)
- **Student:** This LNN (CfC architecture)
- **Method:** Teacher generates probability distributions on many time-series datasets. Student learns to match those distributions.
- **Why:** Teacher models are 700M+ params and too slow for real-time use. The LNN student can run in microseconds.

This requires the direct-training approach (current project) to work first, proving the LNN architecture can learn real time-series patterns.

---

## Pipeline Projects

### 1. Starlink Orbital Decay Prediction
- **What:** Predict SpaceX Starlink satellite altitude loss during geomagnetic storms
- **Source:** CelesTrak TLE data
- **Frequency:** Daily
- **LNN angle:** Famous test case — 40 Starlinks burned up in Feb 2022 storm. Can LNN predict which storms kill satellites?
- **URL:** https://celestrak.org/
- **Dependency:** Builds directly on the current storm prediction model

### 2. Satellite Drag from Solar Activity
- **What:** Predict atmospheric density changes and satellite altitude loss during storms
- **Source:** JB2008 atmospheric model / Space-Track.org
- **Frequency:** Per orbit
- **LNN angle:** Direct application — solar activity input → drag prediction output
- **URL:** https://sol.spacenvironment.net/

### 3. Space Debris Count Forecasting
- **What:** Predict growth of tracked orbital debris objects over time
- **Source:** ESA DISCOS (Database and Information System Characterising Objects in Space)
- **Frequency:** Monthly
- **LNN angle:** Long-term trend prediction with irregular reporting intervals
- **URL:** https://discosweb.esoc.esa.int/

### 4. Satellite Conjunction (Near-Miss) Alerts
- **What:** Predict close approach probability between satellites/debris
- **Source:** Space-Track.org (18th Space Defense Squadron)
- **Frequency:** Real-time
- **LNN angle:** Irregular event timing, multi-variate orbital mechanics inputs
- **URL:** https://www.space-track.org/

### 5. Mars Weather Prediction
- **What:** Predict temperature, pressure, wind at Jezero Crater from Perseverance rover
- **Source:** NASA PDS (Planetary Data System)
- **Frequency:** Daily (with transmission delay)
- **LNN angle:** Irregular timestamps due to relay scheduling
- **URL:** https://pds-atmospheres.nmsu.edu/

### 6. Near-Earth Asteroid Close Approaches
- **What:** Predict distance and velocity of passing asteroids
- **Source:** NASA CNEOS (Center for Near-Earth Object Studies) API
- **Frequency:** Updated as discovered
- **LNN angle:** Irregular discovery cadence, multi-variate trajectory data
- **URL:** https://cneos.jpl.nasa.gov/

### 7. Cosmic Ray Intensity
- **What:** Predict neutron monitor counts (anticorrelated with solar activity)
- **Source:** NMDB (Neutron Monitor Database)
- **Frequency:** 1-hourly
- **LNN angle:** Inverse relationship to solar cycle, good for multi-input models
- **URL:** https://www.nmdb.eu/

### 8. Voyager 1/2 Interstellar Particle Data
- **What:** Predict particle flux measurements from the edge of the solar system
- **Source:** NASA CDAWeb (Coordinated Data Analysis Web)
- **Frequency:** Daily
- **LNN angle:** Extremely irregular data gaps (signal takes 22+ hours to reach Earth), unique physics
- **URL:** https://cdaweb.gsfc.nasa.gov/

---

## Priority Order
1. Solar Wind → Kp/Dst **(current — v4 ready for GPU training)**
2. Starlink Orbital Decay (dramatic, builds on #1)
3. Satellite Drag (extends #1 to satellite ops)
4. Distillation from foundation models (the research paper goal)
5. Cosmic Ray Intensity (clean data, good benchmark)
6. Mars Weather (unique, attention-grabbing)
7. Near-Earth Asteroids (public interest, NASA API)
8. Space Debris Count (slow data, long-term)
9. Satellite Conjunction Alerts (complex, needs account)
10. Voyager 1/2 (coolest dataset, hardest to work with)
