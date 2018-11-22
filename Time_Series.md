# Time Series

<br>

Time series problems are intrinsically dynamic and moving. 
This amplifies the sensitivity to overfitting and can also make it challenging for some models to find predictive signals to begin with.

<br>

---
### Time Series Features


1. Date Time Features : These are components of the time step itself for each observation. <br>

> ex) Month, Day, ... <br>

2. Lag Features : These are values at prior time steps. <br>

> ex) Value(t-1), Value(t+1), ... <br>

3. Window Features : These are a summary of values over a fixed window of prior time steps. <br>

> ex) Rolling(mean, max, min,...),...
