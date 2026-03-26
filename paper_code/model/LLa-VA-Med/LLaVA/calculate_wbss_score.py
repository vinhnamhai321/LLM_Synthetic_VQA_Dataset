from nltk.corpus import wordnet
import re
def calculate_wbss_score(s1: str, s2: str):
    pass


# String need to be cleaned

s = "Text    cleaning? Regex-based!??... Works fine...      "
# Clear punctuation
s_clean = re.sub(r'[^\w\s]', '', s)
print(s_clean)
