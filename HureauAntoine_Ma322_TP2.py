# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:02:06 2021

@author: carca
"""

from matplotlib import pyplot as plt
from math import sqrt,pi,cos,sin
import numpy as np
from scipy.integrate import odeint, solve_ivp

#4.2.2 Résolution numérique de l'équation exacte, par la méthode d'Euler

g=9.81
L=1
h=0.04
t=np.arange(0,4,h)
Theta=list()
Thetap=list()
def pendule(Y,t):
    return np.array([Y[1],(-g/L)*sin(Y[0])])

for i in range (len(t)) :
    Theta.append((pi/2)*cos(sqrt(g/L)*t[i]))
    Thetap.append(-(pi/2)*sqrt(g/L)*sin(sqrt(g/L)*t[i]))

def euler_explicite(f,y_0,h):
    n=100
    Ye=np.zeros((n,y_0.size))
    Ye[0,:]=y_0.reshape(2)
    for k in range (n-1):
        Ye[k+1,:]=Ye[k,:]+h*f(Ye[k,:],t[k])
    return (Ye,t)

y0=np.array([[pi/2],[0]])
Y_1,t=euler_explicite(pendule,y0,h)

#4.2.3 Résolution numérique de l'équation exacte, par la méthode de Runge Kutta
#d'ordre 4

def range_kutta(f,y_0,h):
    n=100
    Ye=np.zeros((n,y_0.size))
    Ye[0,:]=y_0.reshape(2)
    for k in range(n-1):
        k1=f(Ye[k,:],t[k])
        k2=f(Ye[k,:]+(h/2)*k1,t[k]+(h/2))
        k3=f(Ye[k,:]+(h/2)*k2,t[k]+(h/2))
        k4=f(Ye[k,:]+h*k3,t[k]+h/2)
        Ye[k+1,:]= Ye[k,:] + (h/6)*( k1+ 2*k2 + 2*k3 + k4)
    return(Ye,t)

y0=np.array([[pi/2],[0]])
Y_2,t=range_kutta(pendule,y0,h)

#4.2.4 Résolution numérique de l'équation exacte, avec le solveur odeint

Yode = odeint(pendule,[pi/2,0],t)
def pendule2(t,Y):
    r=Y[1]
    p=Y[0]
    dr=(-g/L)*sin(p)
    return np.array([r,dr])
t_span = np.linspace(0, 4, num=100) 
sol = solve_ivp(pendule2,[0,4],[pi/2,0],t_eval=t_span)
#Tracé des résultats obtenus
plt.subplot(211)
plt.plot(t,Theta,'b',label="Equation lineaire")
plt.plot(t,Y_1[:,0],'r',label="Euler")
plt.plot(t,Y_2[:,0],'g',label="Range Kutta")
plt.plot(t,Yode[:,0],'y',label="Odeint")
plt.plot(sol.t,sol.y[0,:],'purple',label="Solveur ivp")
plt.title("Theta(t) ")
plt.xlabel("t en secondes")
plt.ylabel("Theta(t)")
plt.legend()
plt.grid()
plt.show()

#Tracé des portraits de phase
plt.plot(Theta,Thetap,'b',label="Linéaire")
plt.plot(Y_1[:,0],Y_1[:,1],'r',label="Euler")
plt.plot(Y_2[:,0],Y_2[:,1],'g',label="Range Kutta")
plt.plot(Yode[:,0],Yode[:,1],'y',label="Odeint")
plt.title("Portrait de phase")
plt.xlabel("Thetap(t)")
plt.ylabel("Theta(t)")
plt.legend()
plt.grid()
plt.show()

#6.2.1 Résolution numérique
M1=15
M2=200
C2=1200
K1=50000
K2=5000

def suspension(Y,t):
    return np.array([ Y[2], Y[3], (1/M1)*(C2*(-Y[2]+Y[3])-(K1+K2)*Y[0]+K2*Y[1]), (1/M2)*(C2*(Y[2]-Y[3])+K2*(Y[0]-Y[1])-1000)])
Y0=[0,0,0,0]
u=np.arange(0,5,0.1)
Yo = odeint(suspension,[0,0,0,0],u)
plt.plot(u,Yo[:,1],'r',label="x2(t)")
plt.plot(u,Yo[:,0],'b',label="x1(t)")
plt.title("Affaissement de la roue et de la caisse")
plt.xlabel("x(t)")
plt.ylabel("t")
plt.legend()
plt.grid()
plt.show()