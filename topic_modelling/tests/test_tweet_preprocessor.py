from topic_modelling.preprocessor_tweet import tweet_preprocessor as clean
from topic_modelling.preprocessor_tweet import remove_emoji, demoji_from_text

"""
The tweet_preprocessor method should remove:URL,MENTION, RESERVED,SMILEY,EMOJI
"""


test_cases = [
    # Retweets and mentions
    {'in':'RT @StylishRentals: Love this! "Palm Springs Mid century Modern - Houses for Rent in Palm Springs" @airbnb #Travel https://t.co/rzP2YB9k7t',
     'out':': Love this! "Palm Springs Mid century Modern - Houses for Rent in Palm Springs" #Travel'},
    {'in':'RT @hocais: #Rize #Turkey #Ayder #CityOfAllSeasons #HerMevsiminKenti #Travel Görmelisin https://t.co/6gJQObKA8y',
     'out':': #Rize #Turkey #Ayder #CityOfAllSeasons #HerMevsiminKenti #Travel Görmelisin'},
    # Mutlible urls
    {'in':'#travel GOODBYE AUSTRALIA! 625 Days of Travel Compilation Video - https://t.co/YUR1k0hyIv #RT #Retweet https://t.co/eh6D2dvUTW',
     'out':'#travel GOODBYE AUSTRALIA! 625 Days of Travel Compilation Video - #RT #Retweet'},
    # Broken Urls
    {'in':'RT @Kayako: The 4 Best Retention Strategies to Reduce Customer Churn \n\nhttps://t.co/1dCwQwQ6i5\n#custserv #loyalty #retention https://t.co/n…',
     'out':': The 4 Best Retention Strategies to Reduce Customer Churn #custserv #loyalty #retention'},
    # Non English text languages
    {'in': 'Capsule Corp archive.\n #jworld #capsulecorp #dragonballz #anime #ikebukuro #池袋 #東京 #日本 #japan #japanese #travel #va… https://t.co/vmjhVJw8JC',
        'out': 'Capsule Corp archive. #jworld #capsulecorp #dragonballz #anime #ikebukuro #池袋 #東京 #日本 #japan #japanese #travel #va…'},
]

emoji_test_cases = [
    {'in': 'RT😲 😲@hocais😲: 😲#Rize😲 #Turkey #Ayder #CityOfAllSeasons #HerMevsiminKenti #Travel Görmelisin 😲https://t.co/6gJQObKA8y',
        'out': 'RT @hocais: #Rize #Turkey #Ayder #CityOfAllSeasons #HerMevsiminKenti #Travel Görmelisin https://t.co/6gJQObKA8y'},
    {'in': '🤔 🙈 me así, se 😌 ds 💕👭👙 hello 👩🏾‍🎓 emoji hello 👨‍👩‍👦‍👦 how are 😊 you today🙅🏽🙅🏽',
        'out': 'me así, se ds hello emoji hello how are you today'},
    {'in': '🎅🏾 going 5strong innings with 5k’s🔥 🐂 🌋🌋 👹  🤡 🚣🏼 👨🏽‍⚖️  🔥🔥 🇲🇽  🇳🇮 🔥🔥!!!',
        'out': 'going 5strong innings with 5k’s !!!'},
    {'in': "🎅I bet you didn't know that 🙋, 🙋‍♂️, and 🙋‍♀️ are three different emojis.",
        'out': "I bet you didn't know that , , and  are three different emojis."},
]

def test_tweet_preprocessor():
    """
    Known edge cases :
    * Görmelisin -> Grmelisin should be fine as long as it's consident across tweets

    """
    for item in test_cases:
        assert (" ".join(clean(item['in']).split())) == " ".join(item['out'].split())


def test_demoji():
    for item in emoji_test_cases:
        assert (" ".join(demoji_from_text(item['in']).split())) == " ".join(item['out'].split())

# def test_remove_emoji():
#     """
#     Doesn't work with multiple emojis such as 👩‍👦‍👦
#     """
#     for item in emoji_test_cases:
#         assert (" ".join(remove_emoji(item['in']).split())) == " ".join(item['out'].split())



if __name__ == '__main__':
    print(clean(test_cases[6]['in']))
    print(demoji_from_text(emoji_test_cases[1]['in']))