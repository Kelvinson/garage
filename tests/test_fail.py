import unittest


class FailingTest(unittest.TestCase):
    def test_fail(self):
        assert False
