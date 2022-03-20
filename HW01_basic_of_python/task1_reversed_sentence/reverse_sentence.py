# definition of the function
def reverse(sentence):
    words = sentence.split(' ')
    rev_sentence = ' '.join(reversed(words))
    print(rev_sentence)

# test of the function
reverse('this is the test')