from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import xlwings as xw

class Nature(Enum):
    CALL = 0 #"CALL"
    PUT = 1 #"PUT"

class Type(Enum):
    EUROPEAN = 0 #"EUROPEAN"
    AMERICAN = 1 #"AMERICAN"

class Periode(Enum):
    YEARLY = 365
    MONTHLY = 30

@dataclass
class Option:
    maturity: datetime
    nature: Nature # Call/Put
    type: Type # Europénne / Américaine
    K: float

@dataclass
class Market:
    r: float # Rendement
    vol: float # Volatilité
    div: float # Dividende
    start_date: datetime
    div_date: datetime # Date de dividende
    stock_price: float # Prix de l'actif

@dataclass
class Node:
    stock: float
    pg: float = None # proba gauche
    pm: float = None # proba milieu
    pd: float = None # proba droite
    ptot: float = 0 # proba total
    fg: 'Node' = None # fils gauche
    fm: 'Node' = None # fils milieu
    fd: 'Node' = None # fils droit
    fh: 'Node' = None # frere haut
    fb: 'Node' = None # frere bas
    payoff: float = None

    def _isCloser(self,alpha): #  si il l'optimum pour être un noeud milieu donné
        print((self.stock+alpha*self.stock)/2)
        print((self.stock+self.stock/alpha)/2)
    
    def notCloser(self, alpha, target):
        if (self.stock+self.stock/alpha)/2 <= target:
            return False
        else:
            return True
    
    def test(self):
        aux = self.ptot
        noeud = self.fh
        while noeud != None:
            aux += noeud.ptot
            noeud = noeud.fh
        noeud = self.fb
        while noeud != None:
            aux += noeud.ptot
            noeud = noeud.fb
        return aux

@dataclass
class Node_trunk(Node):
    fp: 'Node' = None # le pere

class Tree:
    n: float
    r : float
    alpha: float = None
    delta: float = None
    mult_std: float = np.sqrt(3)
    today: datetime
    root: Node = None

    def __init__(self, n:int, mark:Market, opt:Option, pruning:list): # Pour créer un arbre, on a besoin du nombre d'itérations, du marché et de l'option
        self.today = mark.start_date
        self.n = n
        self.r = mark.r
        self.delta = ((opt.maturity-self.today).days/n)/Periode.YEARLY.value
        self.alpha = np.exp(mark.r*self.delta+mark.vol*self.mult_std*np.sqrt(self.delta))

        if pruning[0]:
            treshold = pruning[1]
        else:
            treshold = 0 # ie pas de pruning
        n_div = n - np.ceil((mark.div_date-self.today).days*n/(opt.maturity-self.today).days) + 1 # +-1 ?
        div = 0
        rate = np.exp(self.r*self.delta)

        def compute_proba(noeud, r, delta, vol, alpha) -> None: # Pour un noeud donné renvoie les probas des noeuds qui lui sont associées
            """
            noeud.pd = (
                (noeud.fm.stock**(-2)*(noeud.stock**2*np.exp(2*r*delta)*(np.exp(vol**2*delta)-1) +
                noeud.fm.stock**2)-1-(alpha+1)*((noeud.fm.stock**(-1))*noeud.fm.stock-1)) / 
                ((1-alpha)*(alpha**(-2)-1))
            )
            """
            fwd = noeud.stock*rate - div
            noeud.pd = (
                (noeud.fm.stock**(-2)*(noeud.stock**2*np.exp(2*r*delta)*(np.exp(vol**2*delta)-1) +
                fwd**2)-1-(alpha+1)*(fwd/noeud.fm.stock-1)) / 
                ((1-alpha)*(alpha**(-2)-1))
            )

            noeud.pg = (fwd/noeud.fm.stock-1-noeud.pd*((alpha**-1)-1))/(alpha-1)
            noeud.pm = 1-noeud.pd-noeud.pg
            noeud.fm.ptot = noeud.fm.ptot + noeud.ptot * noeud.pm
            noeud.fg.ptot = noeud.fg.ptot + noeud.ptot * noeud.pg
            noeud.fd.ptot = noeud.fd.ptot + noeud.ptot * noeud.pd
            return None # Tout se passe par réference
        
        def target_node(node, target) -> Node: #A un niveau donné, en parcourant la largeur, on trouve le noeud dont le stock est le plus proche de la cible donné
            while node.notCloser(self.alpha, target):
                if node.fb != None:
                    node = node.fb
                else:
                    return node
            return node

        def make_node(noeud, r, delta, alpha, vol) -> None: # On considère ici qu'on a déja trouvé le noeud milieu suivant et qu'on doit créer les noeuds connexes
            if noeud.ptot > treshold:
                if noeud.fm.fb != None: # Il existe deja un noeud en dessous
                    noeud.fd = noeud.fm.fb
                else: # Il n'y a pas de noeud plus bas, il faut le creer
                    noeud.fd = Node(noeud.fm.stock/alpha)
                    #print(noeud.fd.stock)
                    noeud.fm.fb = noeud.fd
                    noeud.fd.fh = noeud.fm
                if noeud.fm.fh != None: # Il existe deja un noeud au dessus
                    noeud.fg = noeud.fm.fh
                else: # Il n'y a pas de noeud plus bas, il faut le creer
                    noeud.fg = Node(noeud.fm.stock*alpha)
                    #print(noeud.fg.stock)
                    noeud.fm.fh = noeud.fg
                    noeud.fg.fb = noeud.fm
                compute_proba(noeud, r, delta, vol, alpha)
            else:
                noeud.fd = noeud.fm 
                noeud.fg = noeud.fm
                noeud.pm = 1
                noeud.pd = 0
                noeud.pg = 0
                noeud.fm.ptot = noeud.fm.ptot + noeud.ptot * noeud.pm
            
        noeud = Node_trunk(mark.stock_price) # La racine de notre arbre
        noeud.ptot = 1 # On note 
        while n != 0: # On parcourt le nombre d'iterations (la hauteur de l'arbre) et on va renvoyer la racine de notre arbre à la fin
            if n != n_div:
                noeud.fm = Node_trunk(noeud.stock*rate) # Valeur du forward
                div = 0
            else:
                noeud.fm = Node_trunk(noeud.stock*rate-mark.div)
                div = mark.div
            noeud.fg = Node(noeud.fm.stock*self.alpha)
            noeud.fd = Node(noeud.fm.stock/self.alpha)
            noeud.fm.fp = noeud
            noeud.fm.fh = noeud.fg # On fait la connexion entre les noeuds
            noeud.fm.fb = noeud.fd
            noeud.fg.fb = noeud.fm
            noeud.fd.fh = noeud.fm
            compute_proba(noeud, mark.r, self.delta, mark.vol, self.alpha)
            aux_noeud = noeud.fh
            while aux_noeud != None: # On construit les noeuds en allant vers le haut
                aux_noeud.fm = target_node(aux_noeud.fb.fg, aux_noeud.stock) # la feuille plus bas a forcement un noeud sur l'autre instance
                make_node(aux_noeud, mark.r, self.delta, self.alpha, mark.vol)
                aux_noeud = aux_noeud.fh
            aux_noeud = noeud.fb
            while aux_noeud != None: # On construit les noeuds en allant vers le nas
                aux_noeud.fm = target_node(aux_noeud.fh.fm, aux_noeud.stock) # la feuille plus haute a forcement un noeud sur l'autre instance
                make_node(aux_noeud, mark.r, self.delta, self.alpha, mark.vol)
                aux_noeud = aux_noeud.fb
            n = n-1 # On a terminé la creation sur la largeur, on peut avancer sur la hauteur
            noeud = noeud.fm
        while noeud.fp != None: # On cherche à revenir à la racine
            noeud = noeud.fp
        self.root = noeud

    def pricing(self, opt:Option):

        def put_call_payoff(nat:Nature, K:float):
            def call(S:float)->float:
                return max(0, S-K)
            def put(S:float)->float:
                return max(0, K-S)
            
            if nat == Nature.CALL:
                return call
            else:
                return put

        def put_call_american_payoff(nat:Nature, K:float):
            def call(P:float, S:float)->float:
                return max(P, S-K)
            def put(P:float, S:float)->float:
                return max(P, K-S)
            
            if nat == Nature.CALL:
                return call
            else:
                return put

        def pricing_european(noeud:Node, put_call_pay_off)->float:

            while noeud.fm != None:
                noeud = noeud.fm
            payoff = noeud.ptot*put_call_pay_off(noeud.stock)
            aux_noeud = noeud.fh
            while aux_noeud != None:
                payoff += aux_noeud.ptot*put_call_pay_off(aux_noeud.stock)
                aux_noeud = aux_noeud.fh
            aux_noeud = noeud.fb
            while aux_noeud != None:
                payoff += aux_noeud.ptot*put_call_pay_off(aux_noeud.stock)
                aux_noeud = aux_noeud.fb
            return payoff*np.exp(-self.r*self.n*self.delta)
        
        def princing_american(noeud:Node, put_call_pay_off, put_call_american_pay_off)->float:

            aux = np.exp(-self.r*self.delta)

            def compute_largeur(node:Node):
                node.payoff = put_call_american_pay_off((node.pg*node.fg.payoff + node.pm*node.fm.payoff + node.pd*node.fd.payoff)*aux, node.stock)
                noeud = node
                while noeud.fh != None:
                    noeud = noeud.fh
                    noeud.payoff = put_call_american_pay_off((noeud.pg*noeud.fg.payoff + noeud.pm*noeud.fm.payoff + noeud.pd*noeud.fd.payoff)*aux, noeud.stock)
                noeud = node
                while noeud.fb != None:
                    noeud = noeud.fb
                    noeud.payoff = put_call_american_pay_off((noeud.pg*noeud.fg.payoff + noeud.pm*noeud.fm.payoff + noeud.pd*noeud.fd.payoff)*aux, noeud.stock)
        
            aux_node = self.root # will be the last truc node
            while aux_node.fm != None:
                aux_node = aux_node.fm
            end_node = aux_node
            end_node.payoff = put_call_pay_off(end_node.stock)
            while end_node.fh != None:
                end_node = end_node.fh
                end_node.payoff = put_call_pay_off(end_node.stock)
            while end_node.fb != None:
                end_node = end_node.fb
                end_node.payoff = put_call_pay_off(end_node.stock)
            
            aux_node = aux_node.fp
            while aux_node.fp != None:
                compute_largeur(aux_node)
                aux_node = aux_node.fp
            return put_call_american_pay_off((aux_node.pg*aux_node.fg.payoff + aux_node.pm*aux_node.fm.payoff + aux_node.pd*aux_node.fd.payoff)*aux, aux_node.stock)

        if opt.type == Type.EUROPEAN:
            return pricing_european(self.root, put_call_payoff(opt.nature, opt.K))
        else: # Option américaine
            return princing_american(self.root, put_call_payoff(opt.nature, opt.K), put_call_american_payoff(opt.nature, opt.K))
        
    def show_attr(self) -> None: # Permet de voir les attributs de notre arbre
        print(
            "\nn", self.n,
            "\nalpha", self.alpha,
            "\ndelta", self.delta,
            "\nmult_std", self.mult_std,
            "\ntoday", self.today
        )

    def print_xl(self) -> None: # Permet d'imprimer sur excel les caractéristiques de l'arbre
        arrond = 6 # nombre de chiffre retenu pour l'arrondi
        wb = xw.Book("tree_debug.xlsx")  # this will open a new workbook
        sheet = xw.sheets.active
        sheet.clear()
        root = self.root
        noeud = root

        # Calcul la largeur de l'arbre maximal pour l'affichage
        while noeud.fm != None:
            noeud = noeud.fm
        n = 1
        aux = noeud
        while aux.fh != None:
            aux = aux.fh
            n +=1
        aux = noeud
        while aux.fb != None:
            aux = aux.fb
            n +=1

        n_largeur = 3*n + n-1
        x,y = 0,int(np.ceil(n_largeur/2))
        noeud = root

        for i in range(self.n+1):
            y_aux = y
            sheet[y_aux,x].value = round(noeud.stock,arrond)
            if noeud.pm != None:
                sheet[y_aux-1,x+1].value = round(noeud.pg,arrond)
                sheet[y_aux,x+1].value = round(noeud.pm,arrond)
                sheet[y_aux+1,x+1].value = round(noeud.pd,arrond)
            aux = noeud
            while aux.fh != None:
                aux = aux.fh
                y_aux -= 4
                sheet[y_aux,x].value = round(aux.stock,arrond)
                if aux.pm != None:
                    sheet[y_aux-1,x+1].value = round(aux.pg,arrond)
                    sheet[y_aux,x+1].value = round(aux.pm,arrond)
                    sheet[y_aux+1,x+1].value = round(aux.pd,arrond)
            y_aux = y
            aux = noeud
            while aux.fb != None:
                aux = aux.fb
                y_aux += 4
                sheet[y_aux,x].value = round(aux.stock,arrond)
                if aux.pm != None:
                    sheet[y_aux-1,x+1].value = round(aux.pg,arrond)
                    sheet[y_aux,x+1].value = round(aux.pm,arrond)
                    sheet[y_aux+1,x+1].value = round(aux.pd,arrond)
            x = x+3
            noeud = noeud.fm
