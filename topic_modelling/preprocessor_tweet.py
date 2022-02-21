import preprocessor as tp
import emoji
import demoji
import re

# @timing
def tweet_preprocessor(text: str) -> str:
    """
    Removes @mentions, #hashtags. URLs, reserved words (RT, FAV), emojis and Smiley faces
    https://github.com/s/preprocessor
    """
    tp.set_options(
        tp.OPT.URL,
        tp.OPT.MENTION,
        tp.OPT.RESERVED,
        # tp.OPT.HASHTAG,
        # tp.OPT.SMILEY,
        # tp.OPT.EMOJI,
        # tp.OPT.NUMBER,
    )
    return tp.clean(text)

def demoji_from_text(text:str) -> str:
    """
    https://pypi.org/project/demoji/
    """
    return demoji.replace(text)

#
# def remove_emoji(text):
#     """
#     https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
#     Will return additional space
#     Won't work for some emojis
#     """
#     regrex_pattern = re.compile(pattern="["
#                                         u"\U0001F600-\U0001F64F"  # emoticons
#                                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                                         "]+", flags=re.UNICODE)
#     return regrex_pattern.sub(r'', text)
#
# def remove_emojis(text):
#     """
#     https://pypi.org/project/emoji/
#     """
#     emoji_list = []
#     data = re.findall(r'\X', text)
#     for word in data:
#         if any(char in emoji.UNICODE_EMOJI['en'] for char in word):
#             emoji_list.append(word)
#
#     return emoji_list



if __name__ == '__main__':
    assert(tp.clean('Preprocessor is #awesome 👍 https://github.com/s/preprocessor')) == 'Preprocessor is'
    assert(remove_emoji('Preprocessor is #awesome 👍 https://github.com/s/preprocessor')) == 'Preprocessor is #awesome  https://github.com/s/preprocessor'

    line = "🎅I bet you didn't know that 🙋, 🙋‍♂️, and 🙋‍♀️ are three different emojis."

    print(demoji_from_text(line))