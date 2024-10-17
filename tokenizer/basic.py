from .base import Tokenizer, get_stats, merge_vocab

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size):
        """
        Generate merges: (Dict[tuple(str, str), str]), and vocab: (Dict[str, int])
        Args:
            text (List[str]): all the input texts
        """
        vocab, reverse_vocab = self.initialize_vocab(text)
        print(f'inital vocab size: {len(vocab)}')
        print(vocab)
        num_merge = vocab_size - len(vocab)
        tokens = self.tokenize_text(text)
        merges = {}

        for i in range(num_merge):
            stats = get_stats(tokens)
            # save two ids for padding and UNK
            next_id = len(vocab) + 2 
            pair = max(stats, key=stats.get)
            merges[pair] = pair[0] + pair[1]
            vocab[next_id] = merges[pair]
            reverse_vocab[merges[pair]] = next_id
            tokens = merge_vocab(tokens, pair)
        self.merges = merges
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
    
    def initialize_vocab(self, text):
        """
        This is a character based vocabulary 
        Args:
            text (List[str]): all  training  texts
        Returns:
            Dict[int, string]: id-token map
        """
        unique_characters = list(set("".join(text)))
        vocab = {}
        reverse_vocab = {}
        for i, c in enumerate(unique_characters):
            # we save index 0 for padding token and 1 for UNK
            vocab[i+2] = c
            reverse_vocab[c] = i+2
        return vocab, reverse_vocab
    
    


