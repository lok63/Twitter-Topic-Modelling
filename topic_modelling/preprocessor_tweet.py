import preprocessor as tp

# @timing
def tweet_preprocessor(text: str) -> str:
    """
    Removes @mentions, #hashtags. URLs, reserved words (RT, FAV), emojis and Smiley faces
    https://github.com/s/preprocessor
    """
    tp.set_options(
        tp.OPT.URL,
        tp.OPT.MENTION,
        # tp.OPT.HASHTAG,
        tp.OPT.RESERVED,
        tp.OPT.SMILEY,
        tp.OPT.EMOJI,
        # tp.OPT.NUMBER,
    )
    return tp.clean(text)

if __name__ == '__main__':
    assert(tp.clean('Preprocessor is #awesome 👍 https://github.com/s/preprocessor')) == 'Preprocessor is'
