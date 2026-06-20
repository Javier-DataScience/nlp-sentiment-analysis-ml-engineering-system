# ============================================================
# VOCABULARY MODULE (STABLE NLP INTERFACE)
# ------------------------------------------------------------
# PURPOSE:
# Maps tokens ↔ integer IDs for NLP models.
#
# Provides:
# - token → id mapping
# - id → token mapping
# - safe fallback handling
# ============================================================


class Vocabulary:

    def __init__(self):

        self.token_to_id = {}
        self.id_to_token = {}

        # Reserve special token for unknown words
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.add_token(self.pad_token)
        self.add_token(self.unk_token)

    def add_token(self, token):

        if token not in self.token_to_id:

            idx = len(self.token_to_id)

            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def build(self, token_list):

        for token in token_list:
            self.add_token(token)

    # ========================================================
    # CRITICAL FIX: compatibility method
    # ========================================================
    def get(self, token):

        return self.token_to_id.get(token, self.token_to_id[self.unk_token])

    def decode(self, idx):

        return self.id_to_token.get(idx, self.unk_token)