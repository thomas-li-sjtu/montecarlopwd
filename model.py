import abc


class Model(metaclass=abc.ABCMeta):
    """Abstract base class for models.

    For each probability p, we handle its logprob -- i.e., with some
    abuse of notation, the base-2 logarithm changed of sign:
    -math.log2(p).
    """

    @abc.abstractmethod
    def generate(self, maxlen=100):
        """Generate a random password according to the model.

        Returns (logprob, passwd); passwd is the random password and
        logprob is its probability.
        """
        pass

    def sample(self, n, maxlen=100):
        """Generate a sample of n passwords."""
        return (self.generate(maxlen=maxlen) for _ in range(n))

    @abc.abstractmethod
    def logprob(self, word):
        """Return the logprob of word according to the model."""
        pass
