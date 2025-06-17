# Custom exception for when the LP DAAC server is unreachable.
class LPDAACServerUnreachable(Exception):
    pass

# Custom exception for when the output is blank (e.g., all NaN values).
class BlankOutputError(Exception):
    pass
