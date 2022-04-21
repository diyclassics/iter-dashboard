import re
import unicodedata
from cltk.alphabet.lat import JVReplacer

replacer = JVReplacer()

# Helper function for preprocessing
def preprocess(text, lower=True, normalize=True, punctuation=False, numbers=False, remove_spaces=False, diacriticals=True):
    if lower:
        text = text.lower() # Lowercase

    if normalize:
        text = replacer.replace(text)

    if not punctuation:
        # Remove punctuation
        punctuation ="\"#$%&\'()*+,/:;<=>@[\]^_`{|}~.?!«»—“-”"
        misc = '¡£¤¥¦§¨©¯°±²³´µ¶·¸¹º¼½¾¿÷·–‘’†•ↄ∞⏑〈〉（）'
        misc += punctuation
        translator = str.maketrans({key: " " for key in misc})
        text = text.translate(translator)

    if not numbers:
        # Remove numbers
        translator = str.maketrans({key: " " for key in '0123456789'})
        text = text.translate(translator)

    if remove_spaces:
        text = "".join(text.split())

    if not diacriticals:
        text = remove_diacriticals(text)

    # Fix spacing
    text = re.sub(' +', ' ', text)

    text = unicodedata.normalize('NFC', text)

    return text.strip()