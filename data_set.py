class DataSet:
    # Inicializace
    def __init__(self, rows=0, columns=0):
        self.patterns = []
        self.rows = rows
        self.columns = columns
        self.num_neurons = rows * columns

    # Přidání patternu
    def add_pattern(self, pattern):
        self._check_pattern_size(pattern)
        pattern = self._normalize_pattern(pattern)
        self.patterns.append(pattern)
        
    # Kontrola velikosti patternu
    def _check_pattern_size(self, pattern):
        if len(pattern) != self.rows or len(pattern[0]) != self.columns:
            raise ValueError("Pattern size does not match the training set size")
        
    def _normalize_pattern(self, pattern):
        return [[1 if cell == 1 else -1 for cell in row] for row in pattern]
    
    # Převod -1 na 0 v pattern a rozdělení pattern do řádků podle počtu sloupců
    def get_pattern(self, index):
        pattern = self.patterns[index]
        return [[1 if cell == 1 else 0 for cell in row] for row in pattern]
    
    def display_patterns(self):
        for pattern in self.patterns:
            self.display_pattern(pattern)

    def display_pattern(self, pattern):
        width = len(pattern[0])
        print('+' + '-' * width + '+')
        for row in pattern:
            print('|' + ''.join(['*' if cell == 1 else ' ' for cell in row]) + '|')
        print('+' + '-' * width + '+')