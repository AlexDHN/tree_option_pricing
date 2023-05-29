from arbre_trino import Option, Market, Nature, Tree
import numpy as np
from statistics import NormalDist
from datetime import datetime, timedelta

class BS:
    S: float # sous jacent
    K: float # strike
    sigma: float # volatilité implicite
    T: float # echéance de l'option
    r: float # taux d'intéret
    q: float # taux de dividend
    pricing: float # Le pricing de l'option
    delta:float # le Delta
    gamma:float # Le gamma
    theta:float # Le theta
    rho:float # Le rho
    vega:float # Le vega

    def __init__(self, mark:Market, opt:Option):

        self.S = mark.stock_price
        self.K = opt.K
        self.sigma = mark.vol
        self.T = ((opt.maturity-mark.start_date).days)/365
        self.r = mark.r
        if mark.div == 0:
            self.q = 0
        else:
            #self.q = 0.02 # Aribtrairement à 2%
            self.q = mark.div/self.S

        d1 = (1/(self.sigma*np.sqrt(self.T)))*(np.log(self.S/self.K) + (self.r + 0.5*self.sigma**2)*self.T)
        d2 = d1 - self.sigma*np.sqrt(self.T)
        N = NormalDist()

        def densite(x):
            return np.exp(-x**2/2)*(1/np.sqrt(2*np.pi))

        if opt.nature == Nature.CALL:
            self.pricing = self.S*np.exp(-self.q*self.T)*N.cdf(d1) - np.exp(-self.r*self.T)*self.K*N.cdf(d2)
            self.delta = np.exp(-self.q*self.T)*N.cdf(d1)
            self.theta = -np.exp(-self.q*self.T)*(self.S*densite(d1)*self.sigma)/(2*np.sqrt(self.T))-self.r*self.K*np.exp(-self.r*self.T)*N.cdf(d2) + self.q*self.S*np.exp(-self.q*self.T)*N.cdf(d1)
            self.rho = self.K*self.T*np.exp(-self.r*self.T)*N.cdf(d2)
        else:
            self.pricing = np.exp(-self.r*self.T)*self.K*N.cdf(-d2) - self.S*np.exp(-self.q*self.T)*N.cdf(-d1)
            self.delta = -np.exp(-self.q*self.T)*N.cdf(-d1)
            self.theta = -np.exp(-self.q*self.T)*(self.S*densite(d1)*self.sigma)/(2*np.sqrt(self.T))+self.r*self.K*np.exp(-self.r*self.T)*N.cdf(-d2) - self.q*self.S*np.exp(-self.q*self.T)*N.cdf(-d1)
            self.rho = -self.K*self.T*np.exp(-self.r*self.T)*N.cdf(-d2)

        self.gamma = np.exp(-self.q*self.T)*densite(d1)/(self.S*self.sigma*np.sqrt(self.T))
        self.vega = self.S*np.exp(-self.q*self.T)*densite(d1)*np.sqrt(self.T)

    def compute_delta_gamma(self,n:int, opt:Option, mark:Market, born:float, pas:float)->float:
        S = mark.stock_price
        X = np.linspace(S-born,S+born,pas)
        eps = X[1] - X[0]
        for i in range(len(X)):
            mark.stock_price = X[i]
            X[i] = Tree(n, mark, opt,[True,1e-9]).pricing(opt)
        X1 = np.diff(X)/eps # dérivée première numérique
        print("Delta arbre:", np.mean(X1))
        X2 = np.zeros(len(X)-2)
        for i in range(1,len(X)-1):
            X2[i-1] = (X[i+1]+X[i-1]-2*X[i])/eps**2 # Dérivée seconde numérique
        print("Gamma arbre:", np.mean(X2))
        return(np.mean(X1), np.mean(X2))

    def compute_theta(self,n:int, opt:Option, mark:Market, delta:float, nb:float)->float: # delta = nombre de jours à ajouter, nb nombre de fois
        PRIX = np.zeros(nb)
        T = np.zeros(nb)
        for i in range(nb):
            tre = Tree(n, mark, opt,[True,1e-9])
            T[i] = tre.delta
            PRIX[i] = tre.pricing(opt)
            mark.start_date = mark.start_date + timedelta(days=delta)
        rep = -np.mean(np.divide(np.diff(PRIX),np.diff(T)))
        print("Theta arbre:", rep)
        return rep
    
    def compute_rho(self,n:int, opt:Option, mark:Market, born:float, pas:float)->float:
        r = mark.r
        X = np.linspace(r-born,r+born,pas)
        eps = X[1] - X[0]
        for i in range(len(X)):
            mark.r = X[i]
            X[i] = Tree(n, mark, opt,[True,1e-9]).pricing(opt)
        X = np.diff(X)/eps
        print("rho arbre:", np.mean(X))
        return(np.mean(X))
    
    def compute_vega(self,n:int, opt:Option, mark:Market, born:float, pas:float)->float:
        sig = mark.vol
        X = np.linspace(sig-born,sig+born,pas)
        eps = X[1] - X[0]
        for i in range(len(X)):
            mark.vol = X[i]
            X[i] = Tree(n, mark, opt,[True,1e-9]).pricing(opt)
        X = np.diff(X)/eps
        print("Vega arbre:", np.mean(X))
        return(np.mean(X))
    
    def print_bs(self):
        print("Pricing:", self.pricing)
        print("Delta:",self.delta)
        print("Gamma",self.gamma)
        print("Vega", self.vega)
        print("Theta",self.theta)
        print("Rho",self.rho)

