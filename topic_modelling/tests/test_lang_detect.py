from topic_modelling.preprocessor_spacy import SpacyPreprocessor
from topic_modelling.preprocessor_all import get_language

spp = SpacyPreprocessor(model='en_core_web_lg')

test_cases = [
    # EN
    {'in': 'RT @Atulmaharaj: #myanmar up close by none other than @DivsiGupta #Travel #blog https://t.co/ItUUuqzPZL',
     'out': 'en'},
    {'in': 'images from #Paris. An apartment, dinner, music, dance, you and me, amazing #travel : https://t.co/Ls414kzh7V https://t.co/oXxYfjSsWi',
     'out': 'en'},
    {'in': 'My latest #blog post is all about #seminyak top dining spots &amp; must-visit #beach bars. #TravelTuesday #Travel… https://t.co/ctB12jOpwd',
     'out': 'en'},
    {'in': 'RT @swenomad: Winter wonderland ⛄🎄☃\nhttps://t.co/HuxACt9Yt0\n#travel #sweden #photography https://t.co/fDb5Xj7ojT',
     'out': 'en'},
    {'in': 'Please RT! #travel #tourism #vacation Trivial pursuits of the mind https://t.co/BojRsioFsR',
     'out': 'en'},
    {'in': 'RT @PRicanFilmmaker: LIVE on #Periscope: \n🎥 #NeydaCam - 🎄🎄🎅🎅 Christmas displays in Rideo Drive, #GoLive #LosAngeles #travel #perisc… https:…',
     'out': 'en'},
    {'in': 'A Spiritual Experience in The City of Light, Varanasi https://t.co/ntrB5Cfw3v #asia #destinations #trave',
     'out': 'en'},
    {'in': "#Hemingway was #dazzled by this scenery of the Ebro at dusk - #Discover #Amposta https://t.co/WJsojAhwWR #travel https://t.co/kmdR8tIkEf",
     'out': 'en'},
    {'in': "Please RT? #travel #traveller 84 130km // ENGELBERG // SWITZERLAND https://t.co/3deSF9TmG",
     'out': 'en'},
    # FR
    { 'in': "'RT @CMGsportsclub: Yoga do Brasil, un havre de paix à l’autre bout du monde https://t.co/h6UNf2tTsa #yoga #bresil #meditation #holiday'",
        'out': 'fr'},
    # ES
    {'in': '¿Me ayudarías a ganar la aventura de mi vida? https://t.co/ZwIeUgG8R0 #visitpanama #panamabestinfluencer #travel #VR',
     'out': 'es'},
    # Ru
    {'in': 'RT @yaroslavlitc: Церковь Спаса на Городу #Ярославль https://t.co/ceCfiSpZWo #photo #travel https://t.co/WIcuKv5TBM',
     'out': 'ru'},
]

def test_lang_detect():
    """
    If it's one word, use the en model
    """
    for test_case in test_cases:
        assert get_language(test_case['in']) == test_case['out']


def test_lang_detect_spacy():
    """
    If it's one word, use the en model
    """
    for test_case in test_cases:
        lang, profile = spp.detect_language_single(test_case['in'])
        assert lang == test_case['out']
