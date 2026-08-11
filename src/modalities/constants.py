"""Constants shared across layers of the codebase.

Only for values that several unrelated modules must agree on. A constant used by one module belongs
in that module.
"""

# The class index that torch.nn.CrossEntropyLoss skips by default. A dataset writes it into the
# positions of its targets that should not contribute to the loss; a metric skips the same
# positions. It lives here rather than in either of them because both a dataset module and a
# metric module need it, and neither should have to import the other to get it.
IGNORE_INDEX = -100
