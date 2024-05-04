import random
import torch
from nltk import tokenize
from nlpaug.augmenter.word import BackTranslationAug

device = "cuda" if torch.cuda.is_available() else "cpu"


# from paper ESACL we can know use random_delete and random_swap will be get better score
# and know change 3 sentences will be better to xsum dataset
class DocumentAugmentation():
    """
    Document Augmentation Approaches
    """

    def __init__(self, n, input):
        """
        Initialize the class
        :param n: how many sentences are selected for augmentation
        :param input: input sequence, string
        """
        self.n = n
        self.input = input
        self.sentences = tokenize.sent_tokenize(input)
        self.back_translation_aug = BackTranslationAug(
            from_model_name=r'D:\machine_learning\NLP\Abstract_Summarization_demo2\model\facebook\wmt19-en-de',
            to_model_name=r'D:\machine_learning\NLP\Abstract_Summarization_demo2\model\facebook\wmt19-de-en',
            device=device,
            batch_size=64
        )

    def RandomSwap(self):
        """
        randomly select two sentences in the input document and swap their positions. Do this $n$ times.
        :return:
        """
        self.augmented_sentences = self.sentences
        if len(self.sentences) >= 2:
            for i in range(self.n):
                # location is a list contains two random numbers selected
                location = random.sample(range(len(self.augmented_sentences)), 2)
                sent1 = self.augmented_sentences[location[0]]
                sent2 = self.augmented_sentences[location[1]]
                # swap two sentences
                self.augmented_sentences[location[0]], self.augmented_sentences[location[1]] = sent2, sent1

    def RandomDeletion(self):
        """
        randomly delete n sentences from the input document.
        :return:
        """
        self.augmented_sentences = self.sentences
        # Here we require that the augmented document should have at least one sentence
        if self.n <= len(self.sentences) - 1:
            # location is a list contains two random numbers selected
            location_delete = random.sample(range(len(self.augmented_sentences)), self.n)
            update_sentence = [self.augmented_sentences[i] for i in range(len(self.augmented_sentences)) if
                               i not in location_delete]
            self.augmented_sentences = update_sentence

    def BackTranslation(self):
        sentences = tokenize.sent_tokenize(self.input)
        augs = ""
        update_sentence = self.back_translation_aug.augment(sentences)
        for text in update_sentence:
            augs += text
        self.augmented_sentences = augs

    def BackTranslation_sentence(self, text):
        sentences = tokenize.sent_tokenize(text)
        augs = ""
        update_sentence = self.back_translation_aug.augment(sentences)
        for text in update_sentence:
            augs += text
        return augs
