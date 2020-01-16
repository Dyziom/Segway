import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# nastawy
Kc = -2000 * 1.0
tauI = 1
tauD = 2

# definicja modelu Segwaya
def segway(s):
    R = 0.483/2
    M = 47.3
    m = 63.968
    J = 73.5041
    l = 1.0400
    L = 0.0513
    kT = 0.75
    kbemf = 0.5
    ra = 1.4
    c1 = 0.01
    c2 = 0.01
    g = 9.81
    a = kT/ra
    b = kT*kbemf/(R*ra)
    A = -a/J
    B = m*g/J
    g1 = a/(R*(m+M))
    g2 = (b/R+c2)/(M+m)
    g3 = (b*g1-a*g2)/(-a)
    g4 = c1/J
    g5 = (M*g*L+m*g*l)/J
    V = g1/s+g2
    d = B / (s * s + g4 * s - g5)
    Theta = (A*s+A*g3) / (s*s*s + (g4+g2)*s*s + (g4*g2-g5)*s - g2*g5)
    Theta = Theta * vin + d * din #TODO
    xdot = np.zeros(3)
    xdot[0] = V
    xdot[1] = Theta
    xdot[2] = d
    return xdot

# Wartości początkowe
V_ss = 1.00
Theta_ss = 1
d_ss = 0.01
x0 = np.empty(3)
x0[0] = V_ss
x0[1] = Theta_ss
x0[2] = d_ss


# Przepływ czasu symulacji
s = np.linspace(0, 5, 1000)

# Zmienna do wykresu
V = np.ones(len(s)) * V_ss
Theta = np.ones(len(s)) * Theta_ss
d = np.ones(len(s)) * d_ss


# Zmienne do obliczeń
op = np.ones(len(s))*Theta_ss  # controller output
pv = np.zeros(len(s))  # process variable
e = np.zeros(len(s))   # error
ie = np.zeros(len(s))  # integral of the error
dpv = np.zeros(len(s))  # derivative of the pv
P = np.zeros(len(s))   # proportional
I = np.zeros(len(s))   # integral
D = np.zeros(len(s))   # derivative
sp = np.zeros(len(s))  # set point
sp[0:100] = 10
sp[100:] = 20

# Ograniczenia
op_hi = 1000
op_lo = 0

pv[0] = V_ss
# loop through time steps    
for i in range(len(s)-1):
    delta_t = s[i+1]-s[i]
    e[i] = sp[i] - pv[i]
    if i >= 1:  # calculate starting on second cycle
        dpv[i] = (pv[i]-pv[i-1])/delta_t
        ie[i] = ie[i-1] + e[i] * delta_t
    P[i] = Kc * e[i]
    I[i] = Kc/tauI * ie[i]
    D[i] = - Kc * tauD * dpv[i]
    op[i] = op[0] + P[i] + I[i] + D[i]
    if op[i] > op_hi:  # check upper limit
        op[i] = op_hi
        ie[i] = ie[i] - e[i] * delta_t # anti-reset windup
    if op[i] < op_lo:  # check lower limit
        op[i] = op_lo
        ie[i] = ie[i] - e[i] * delta_t # anti-reset windup
    ts = [s[i],s[i+1]] #TODO
    u[i+1] = op[i] #TODO
    y = odeint(segway,x0,ts,args=()) #TODO
    Ca[i+1] = y[-1][0] #TODO
    T[i+1] = y[-1][1] #TODO
    x0[0] = Ca[i+1] #TODO
    x0[1] = T[i+1] #TODO
    pv[i+1] = T[i+1] #TODO
op[len(s)-1] = op[len(s)-2] #TODO
ie[len(s)-1] = ie[len(s)-2] #TODO
P[len(s)-1] = P[len(s)-2] #TODO
I[len(s)-1] = I[len(s)-2] #TODO
D[len(s)-1] = D[len(s)-2] #TODO

# Wyniki zapisane do pliku
# Column 1 = czas
# Column 2 = prędkość
# Column 3 = odhylenie Theta
# Column 4 = wyhylenie d
data = np.vstack((s, V, Theta, d)) # vertical stack
data = data.T             # transpose data
np.savetxt('data_wyjściowe.txt', data, delimiter=',')
    
# Plot the results
plt.figure(1)
plt.subplot(4,1,1)
plt.plot(s, V, 'b--', linewidth=3)
#plt.ylabel('Cooling T (K)')
plt.legend(['Prędkość'],loc='best')

plt.subplot(4,1,2)
plt.plot(s,Theta,'g-',linewidth=3)
#plt.ylabel('Ca (mol/L)')
plt.legend(['Odhylenie'],loc='best')

plt.subplot(4,1,3)
plt.plot(s,d,'r--',linewidth=3)
#plt.plot(s,sp,'r--',linewidth=2,label='Set Point')
#plt.plot([0,max(s)],[320+20*0.15,320+20*0.15],'k-',label='15% Overshoot')
#plt.plot([0,max(s)],[320+20*0.1,320+20*0.1],'k--',label='10% Overshoot')
plt.legend(['wyhylenie'],loc='best')

plt.subplot(4,1,4)
plt.plot(s,op,'r--',linewidth=3,label='Controller Output (OP)')
plt.plot(s,P,'g:',linewidth=2,label='Proportional (Kc e(t))')
plt.plot(s,I,'b.-',linewidth=2,label='Integral (Kc/tauI * Int(e(t))')
plt.plot(s,D,'k-.',linewidth=2,label='Derivative (-Kc tauD d(PV)/dt)')
plt.legend(loc='best')
plt.ylabel('Output')

plt.show()
