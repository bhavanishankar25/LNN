# Future LNN Projects

## Current: Solar Wind → Geomagnetic Storm Prediction
DSCOVR satellite solar wind data → predict Kp/Dst index at Earth (30-60 min ahead)

---

## Pipeline Projects

### 1. Space Debris Count Forecasting
- **What:** Predict growth of tracked orbital debris objects over time
- **Source:** ESA DISCOS (Database and Information System Characterising Objects in Space)
- **Frequency:** Monthly
- **LNN angle:** Long-term trend prediction with irregular reporting intervals
- **URL:** https://discosweb.esoc.esa.int/

### 2. Satellite Conjunction (Near-Miss) Alerts
- **What:** Predict close approach probability between satellites/debris
- **Source:** Space-Track.org (18th Space Defense Squadron)
- **Frequency:** Real-time
- **LNN angle:** Irregular event timing, multi-variate orbital mechanics inputs
- **URL:** https://www.space-track.org/

### 3. Mars Weather Prediction
- **What:** Predict temperature, pressure, wind at Jezero Crater from Perseverance rover
- **Source:** NASA PDS (Planetary Data System)
- **Frequency:** Daily (with transmission delay)
- **LNN angle:** Irregular timestamps due to relay scheduling, alien atmospheric dynamics
- **URL:** https://pds-atmospheres.nmsu.edu/

### 4. Near-Earth Asteroid Close Approaches
- **What:** Predict distance and velocity of passing asteroids
- **Source:** NASA CNEOS (Center for Near-Earth Object Studies) API
- **Frequency:** Updated as discovered
- **LNN angle:** Irregular discovery cadence, multi-variate trajectory data
- **URL:** https://cneos.jpl.nasa.gov/

### 5. Cosmic Ray Intensity
- **What:** Predict neutron monitor counts (anticorrelated with solar activity)
- **Source:** NMDB (Neutron Monitor Database)
- **Frequency:** 1-hourly
- **LNN angle:** Inverse relationship to solar cycle, good for multi-input models
- **URL:** https://www.nmdb.eu/

### 6. Voyager 1/2 Interstellar Particle Data
- **What:** Predict particle flux measurements from the edge of the solar system
- **Source:** NASA CDAWeb (Coordinated Data Analysis Web)
- **Frequency:** Daily
- **LNN angle:** Extremely irregular data gaps (signal takes 22+ hours to reach Earth), unique physics
- **URL:** https://cdaweb.gsfc.nasa.gov/

### 7. Satellite Drag from Solar Activity
- **What:** Predict atmospheric density changes and satellite altitude loss during storms
- **Source:** JB2008 atmospheric model / Space-Track.org
- **Frequency:** Per orbit
- **LNN angle:** Direct application — solar activity input → drag prediction output
- **URL:** https://sol.spacenvironment.net/

### 8. Starlink Orbital Decay Prediction
- **What:** Predict SpaceX Starlink satellite altitude loss during geomagnetic storms
- **Source:** CelesTrak TLE data
- **Frequency:** Daily
- **LNN angle:** Famous test case — 40 Starlinks burned up in Feb 2022 storm. Can LNN predict which storms kill satellites?
- **URL:** https://celestrak.org/

---

## Priority Order
1. Solar Wind → Kp/Dst (current — proves the LNN works on space weather)
2. Starlink Orbital Decay (dramatic, real-world consequence of storms)
3. Satellite Drag from Solar Activity (extends #1 to practical satellite ops)
4. Cosmic Ray Intensity (clean hourly data, good benchmark)
5. Mars Weather (unique, attention-grabbing for papers/demos)
6. Near-Earth Asteroids (public interest, NASA API is clean)
7. Space Debris Count (slow data, but important long-term problem)
8. Satellite Conjunction Alerts (complex, needs Space-Track account)
9. Voyager 1/2 (coolest dataset but hardest to work with)
