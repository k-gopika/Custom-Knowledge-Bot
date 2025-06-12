# tarot.py
import random

tarot_deck = [
    {
        "name": "The Fool",
        "meanings": {
            "upright": "New beginnings, spontaneity, free spirit, taking a leap of faith.",
            "reversed": "Recklessness, fear of the unknown, foolish behavior, poor judgment."
        }
    },
    {
        "name": "The Magician",
        "meanings": {
            "upright": "Manifestation, resourcefulness, power, inspired action.",
            "reversed": "Manipulation, deception, untapped potential, illusions."
        }
    },
    {
        "name": "The High Priestess",
        "meanings": {
            "upright": "Intuition, subconscious, mystery, inner wisdom.",
            "reversed": "Secrets, withdrawal, blocked intuition, hidden motives."
        }
    },
    {
        "name": "Two of Swords",
        "meanings": {
            "upright": "Indecision, difficult choices, blocked emotions, avoidance.",
            "reversed": "Lies being exposed, confusion, lesser of two evils, no right choice."
        }
    },
    {
        "name": "Ace of Cups",
        "meanings": {
            "upright": "New emotional beginnings, love, compassion, joy.",
            "reversed": "Emotional loss, emptiness, blocked feelings, repressed emotions."
        }
    },
    {
        "name": "Ten of Pentacles",
        "meanings": {
            "upright": "Wealth, family, legacy, long-term success, stability.",
            "reversed": "Loss of legacy, family conflict, instability, broken traditions."
        }
    }
]

def draw_cards(n=3):
    cards = random.sample(tarot_deck, n)
    for card in cards:
        card['orientation'] = random.choice(["upright", "reversed"])
    return cards
