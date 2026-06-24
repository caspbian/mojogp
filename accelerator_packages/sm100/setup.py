from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
