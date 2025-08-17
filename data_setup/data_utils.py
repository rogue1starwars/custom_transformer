import csv
import os


class Vocab:
    """Vocabulary class to handle token-to-index and index-to-token mappings.

    This class provides methods to convert tokens to indices, indices to tokens,
    and to save/load the vocabulary to/from a CSV file.

    Attributes:
        stoi (dict): A dictionary mapping tokens to indices.
        itos (list): A list mapping indices to tokens.
        default_index (int): The index used for out-of-vocabulary tokens.
    """

    def __init__(self, stoi: dict, itos: list, default_index: int):
        """Initialize the Vocab object.
        Args:
            stoi (dict): A dictionary mapping tokens to indices.
            itos (list): A list mapping indices to tokens.
            default_index (int): The index used for out-of-vocabulary tokens.
        """
        self.stoi = stoi
        self.itos = itos
        self.default_index = default_index

    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        """Get the index of a token in the vocabulary."""
        return self.stoi.get(token, self.default_index)

    def lookup_indices(self, token_list: list) -> list:
        """Get the indices of a list of tokens in the vocabulary."""
        return [self.__getitem__(tok) for tok in token_list]

    def get_itos(self) -> list:
        """Get the list of tokens in the vocabulary."""
        return self.itos

    def get_stoi(self) -> dict:
        """Get the dictionary mapping tokens to indices."""
        return self.stoi

    def to_csv(self, filepath: str):
        """Save the vocabulary to a CSV file."""
        if not filepath.endswith(".csv"):
            filepath += ".csv"

        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.exists(filepath):
            print("File already exists. Skipping csv creation")
            return
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "index"])
            for token, index in self.stoi.items():
                writer.writerow([token, index])

    @classmethod
    def from_csv(cls, filepath: str, default_index: int = None) -> "Vocab":
        """Load the vocabulary from a CSV file."""
        stoi = {}
        itos = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                token, index_str = row
                index = int(index_str)
                stoi[token] = index
                # Assuming the CSV maintains the order for itos
                if len(itos) <= index:
                    itos.extend([None] * (index - len(itos) + 1))
                itos[index] = token
        # If default_index is not provided, try to infer it from the vocab, or set to None
        if default_index is None and "<unk>" in stoi:
            default_index = stoi["<unk>"]
        return cls(stoi, itos, default_index)
