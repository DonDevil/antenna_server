class ValidationError(Exception):
    """Raised for invalid input or schema violations."""


class ModelUnavailableError(Exception):
    """Raised when ANN checkpoint is missing or unreadable."""


class ClarificationRequiredError(Exception):
    """Raised when request is too ambiguous to continue safely."""
