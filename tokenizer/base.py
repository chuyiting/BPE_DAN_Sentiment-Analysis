import re, collections

def get_stats(tokens):
    """ 
    Compute the frequency for each pair in the vocab. 

    Args:
        tokens (List[List[str]]): a list of sentences while sentences are represented as lists of tokens
    
    Returns:
        Dict[tuple, int]: All two tuples and their frequencies
    """
    counts = {}
    for t in tokens:
        for pair in zip(t, t[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    
    return counts

def merge_vocab(tokens, pair):
    """
    Merge the pair in the vocabulary

    Args:
        tokens (List[List[str]]): a list of sentences while sentences are represented as lists of tokens
        pair (tuple(str, str)): the most frequent pair of words
       
    Returns:
        (List[List[str]]): The new vocabuary that has the two most frequent pair merged
    """
    new_token = pair[0] + pair[1]
    new_tokens = []
    for s in tokens:
        newids = []
        i = 0
        while i < len(s):
            if i < len(s)-1 and s[i] == pair[0] and s[i+1] == pair[1]:
                newids.append(new_token)
                i += 2 
            else:
                newids.append(s[i])
                i += 1
        new_tokens.append(newids)
    
    return new_tokens


def replace_escape_character(input_string):
    input_string = input_string.replace(' ', r'<space>')
    input_string = input_string.replace('\n', r'<changeline>')
    input_string = input_string.replace('\xa0', r'<nonebreaking>')
    return input_string

def revert_escape_character(input_string):
    input_string = input_string.replace(r'<space>', ' ')
    input_string = input_string.replace(r'<changeline>', '\n')
    input_string = input_string.replace(r'<nonebreaking>', '\xa0')
    return input_string

class Tokenizer: 
    """Base class for tokenizer"""

    def __init__(self):
        # merges is a dictionary of all the merged done during the training process
        # It will be used in the encoding process
        # It has type Dict[tuple(str, str), str]
        self.merges = {}
        # vocab is the map that maps an index to a token
        # It will be used in the decoding process
        # It has type Dict[int, str]
        self.vocab = {}
        # vocab is the map that maps a token to an index
        # It will be used in the encoding process
        # It has type Dict[str, int]
        self.reverse_vocab = {}

    def encode(self, text):
        """
        Turn a string of text into a list of indices 

        Args:
            text (List[str]): the input string 
        Returns:
            (List[List[int]]): list of indices 
        """
        tokens = self.tokenize_text(text)
        while True:
            stats = get_stats(tokens)
            # @#$!^ just an unlikely token; it's a hack TODO refactor this
            pair = min(stats, key=lambda p: self.reverse_vocab.get(self.merges.get(p, '@#$!^'), float('inf')))
            if pair not in self.merges:
                break
            tokens = merge_vocab(tokens, pair)
        
        # default UNK is index 1
        tokens = [[self.reverse_vocab.get(token, 1) for token in sentence] for sentence in tokens]
        return tokens
    
    def decode(self, tokens):
        texts = [[self.vocab.get(token, '<UNK>') for token in sentence] for sentence in tokens]
        texts = [''.join(sentence) for sentence in texts]
        return texts

    def get_vocab_size(self):
        return len(self.vocab)
    
    def tokenize_text(self, text):
        """
        This is a character based tokenization, used in the initializtion phased
        It basically seperate each character and treat each character as a token
        Args:
            text (List[str]): all training texts
        Returns:
            List[List[str]]: list of list of tokens
        """
        tokens = []
        for sentence in text:
            tokens.append(list(sentence))
        return tokens
    
    ###################################################################################
    ############### The following part is modified from Karpathy's work ############### 
    ###################################################################################

    def _build_base_vocab(self, model_file_prefix):
        base_file = model_file_prefix + ".base"
        vocab = {}
        reverse_vocab = {}
        with open(base_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "character based bpm v1"
            # read the merges
            idx = 2 # our index starts from 2
            for line in f:
                # remove the ending change line
                line = line.rsplit('\n', 1) 
                line = ''.join(line) 
                line = revert_escape_character(line)
                vocab[idx] = line
                reverse_vocab[line] = idx
                idx += 1
        return vocab, reverse_vocab


    def _build_vocab(self):
        # You should call _build_base_vocab before calling _build_vocab
        # This method will build on the vocab
        # we have saved 2 indices
        idx = len(self.vocab) + 2
        for _, merged in self.merges.items():
            self.vocab[idx] = merged
            self.reverse_vocab[merged] = idx
            idx += 1
 

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        version_name = "character based bpm v1\n"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write(version_name)
            # the merges dict
            for str1, str2 in self.merges:
                f.write(f"{replace_escape_character(str1)} {replace_escape_character(str2)}\n")
        
        inverted_merges = {merged_token: pair for pair, merged_token in self.merges.items()}

        # we need to save the leaf tokens as well, because we are using character based pairing
        base_file = file_prefix + ".base"
        with open(base_file, "w", encoding="utf-8") as f:
            f.write(version_name)
            for idx, token in self.vocab.items():
                # find the children of this token, if any
                if token in inverted_merges:
                    # if this token has children, skip
                    continue
                else:
                    # otherwise this is leaf token, just print it
                    f.write(f"{replace_escape_character(token)}\n")

        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # find the children of this token, if any
                if token in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    token0, token1 = inverted_merges[token]
                    f.write(f"[{replace_escape_character(token0)}][{replace_escape_character(token1)}] -> [{replace_escape_character(token)}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    f.write(f"[{replace_escape_character(token)}] {idx}\n")

    def load(self, model_file_prefix):
        """Inverse of save() but only for the model file"""
        # read the model file
        merges = {}
        merge_file = model_file_prefix + ".model"
        self.vocab, self.reverse_vocab = self._build_base_vocab(model_file_prefix)

        with open(merge_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "character based bpm v1"
            # read the merges
            for line in f:
                # split by the first space
                try:
                    str1, str2 = line.split()
                    str1 = revert_escape_character(str1)
                    str2 = revert_escape_character(str2)
                    merges[(str1, str2)] = str1 + str2
                except:
                    print("error...")
                    print(line)
        self.merges = merges
        self._build_vocab()
