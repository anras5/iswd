import random
from player import Player


class Nazwisko(Player):

    def __init__(self, name):
        super().__init__(name)
        self.number_opponent_cards = None
        self.cards = []
        self.stack = []

    def startGame(self, cards):
        self.cards = sorted(cards, key=lambda x: x[0])
        self.number_opponent_cards = len(cards)
        self.stack = []

    def takeCards(self, cards_to_take):
        self.cards = sorted(self.cards + cards_to_take, key=lambda x: x[0])
    
    def putCard(self, declared_card):

        # pile is empty
        if declared_card is None:
            self.stack.append(self.cards[0])
            return self.cards[0], self.cards[0]

        for card in self.cards:
            if card[0] >= declared_card[0]:
                if len(self.cards) - self.number_opponent_cards <= 2:
                    self.stack.append(card)
                    return card, card
                else:
                    self.stack.append(cards[0])
                    return self.cards[0], card
        
        self.stack.append(self.cards[0])
        return self.cards[0], (declared_card[0], random.choice([1, 2, 3, 4].remove(declared_card[1])))
    
    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards:
            return True

        if opponent_declaration in self.stack:
            return True

        for card in self.cards:
            if card[0] >= opponent_declaration[0]:
                return False

        return True

    def getCheckFeedback(self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True):
        if not checked and noTakenCards:
            self.number_opponent_cards += noTakenCards
            self.stack = self.stack[:-2] if noTakenCards == 3 else self.stack[:-1]

        if checked and not iDrewCards:
            self.number_opponent_cards += noTakenCards
            self.stack = self.stack[:-1] if iChecked else self.stack[:-2]