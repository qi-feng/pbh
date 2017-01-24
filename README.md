# pbh
Python code that searches for primordial black hole (PBH) evaporation in VERITAS data. It includes:
```
io to the data root file, 
simulations of photons from a point source and uniform background, 
likelihood calculation to test the hypothesis that a list of events are from a point source (a burst candidate), 
seach for burst candidates in a sliding window of a pre-defined duration, 
counting all bursts in data starting from the largest burst, each event is counted once to avoid double counting, 
combining burst search results from individual observations, 
likelihood calculation to determine the rate density of PBH evaporations. 
```
These algorithms are based on Simon Archambault's PhD thesis, and are meant as a secondary analysis for Simon.  
