from re import split

def split_sentences(text):
    parts = split('([.])', text)
    sentences = []

    for part in parts:
        if part == '':
            # remove the empty character after the last split
            continue

        if part == '.' and len(sentences) != 0:
            # add the period to the last added sentence
            sentences[len(sentences) - 1] = sentences[len(sentences) - 1] + '.'

        else :
            sentences.append(part)
    return sentences
