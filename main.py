from arbre_trino import Option, Market, Nature, Type
from bs import BS
from simulation import simulation
from datetime import datetime

start_date = datetime(2022,9,1)
start_price = 100
interest_rate = 0.02
volatility = 0.2
dividend = 3
div_ex_date = datetime(2023,3,1)

maturity = datetime(2023,7,1)
type = Nature.CALL
exercice = Type.AMERICAN
strike = 101

n=500
mark = Market(r = interest_rate, vol = volatility, div = dividend, start_date = start_date, div_date = div_ex_date, stock_price = start_price) # On applique la configuration de notre marché
opt = Option(maturity = maturity, nature = type, type = exercice, K = strike) # La configuration de notre option

prix = simulation(n=n, opt=opt, mark=mark)
print("Black-Scholes")
bs = BS(mark=mark, opt=opt)
bs.print_bs()
print("----------------------------")
print("Arbre")
print("Pricing arbre:", prix)
#delta, gamma = bs.compute_delta_gamma(n,opt,mark,1,10)
#vega = bs.compute_vega(n,opt,mark,0.01,10)
#theta = bs.compute_theta(n,opt,mark,1,10)
#rho = bs.compute_rho(n,opt,mark,0.01,10)