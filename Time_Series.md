# Time Series

<br>

Time series problems are intrinsically dynamic and moving. 
This amplifies the sensitivity to overfitting and can also make it challenging for some models to find predictive signals to begin with.

<br>

Most advanced machine learning algorithms (e.g., XGBoost) are not time-aware. 
They typically look at one row at a time when forming predictions. 
In order to use these methods for forecasting, we need to derive informative features, based on past and present data in time.

<br>

---

### Time Series Framework

<br>

- Feature Derivation Window (FDW):= a rolling window, relative to the Forecast Point, which can be used to derive descriptive features.

- Forecast Window (FW) := the range of future values we wish to predict, called Forecast Distances(FDs).

<br>

---

### Time Series Features


1. Date Time Features : These are components of the time step itself for each observation. <br>

> ex) Month, Day, ... <br>

2. Lag Features : These are values at prior time steps. <br>

> ex) Value(t-1), Value(t+1), ... <br>

3. Window Features : These are a summary of values over a fixed window of prior time steps. <br>

> ex) Rolling(mean, max, min,...),...
