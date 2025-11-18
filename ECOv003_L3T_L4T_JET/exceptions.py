# Custom exception for when the LP DAAC server is unreachable.
class LPDAACServerUnreachable(Exception):
    pass

# Custom exception for when input files are not accessible
class InputFilesInaccessible(Exception):
    pass

# Custom exception for filtering out nighttime data
class DaytimeFilter(Exception):
    pass

# Custom exception for when output data is blank
class BlankOutput(Exception):
    pass

class BlankOutputError(Exception):
    pass
