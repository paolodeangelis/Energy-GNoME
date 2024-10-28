class ImmutableDataError(Exception):
    """Exception raised when attempting to modify immutable data."""

    def __init__(self, message: str):
        super().__init__(message)


class ImmutableRawDataError(Exception):
    """Exception raised when attempting to modify raw data."""

    def __init__(self, message: str):
        super().__init__(message)


class MissingData(Exception):
    """Exception raised when the source data is empty"""

    def __init__(self, message: str):
        super().__init__(message)
