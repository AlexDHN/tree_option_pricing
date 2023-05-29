from arbre_trino import Option, Market, Tree
import time

def simulation(n:int, opt: Option, mark:Market) -> None:
    start = time.time()
    tre = Tree(n, mark, opt,[True,1e-9]) # On créer l'arbre associé à ce marché
    print("Temps de création de l'arbre:",time.time() - start)
    start = time.time()
    print("Pricing:\n----------------------------")
    rep = tre.pricing(opt) #round(tre.pricing(opt),3)
    print(rep)
    print("Temps du pricing:",time.time() - start)
    print("----------------------------")
    return rep