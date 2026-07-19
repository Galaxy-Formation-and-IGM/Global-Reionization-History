### Basic package import

import numpy as np

### src imports



### Defining LX-SFR relation for HMXBs

mean_cx = np.linspace(2.6 * 10 ** (39), 3.7 * 10 ** (39)) #erg s^-1 / M yr^-1, Dijkstra (2012)
mean_cx_scat = mean_cx * np.e ** (1 / 2 * (0.4 * np.log(10)) ** 2)

### New SFRD used in Dijkstra (2012)

def Hopkin_SFRD(z): #Hopkin and Beacom (2006)
    a = 0.017
    b = 0.13
    c = 3.3
    d = 5.3
    return (a + b * z) * h_new / (1 + (z / c) ** d) #in M yr^-1 cMpc^-3