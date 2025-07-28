# Custom exception for when the LP DAAC server is unreachable.
class LPDAACServerUnreachable(Exception):
    pass

from check_distribution import BlankOutputError
