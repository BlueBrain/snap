import bluepysnap._doctools as test_module


class TestClass:
    def __init__(self):
        """TestClass"""
        self.foo_val = 0
        self.bar_val = 1

    def foo(self):
        """foo function for TestClass"""
        return self.foo_val

    def bar(self):
        """bar function for TestClass

        Returns:
            TestClass: return a TestClass
        """
        return self.bar_val

    def foo_bar(self):
        pass


class TestDocUpdater:
    def setup(self):
        self.test_obj = test_module.DocUpdater(TestClass)

    def test_replace_all(self):
        self.test_obj.replace_all("TestClass", "EXPECTED")
        tested = TestClass()
        assert tested.foo.__doc__ == "foo function for EXPECTED"
        expected = """bar function for EXPECTED

        Returns:
            EXPECTED: return a EXPECTED
        """
        assert tested.bar.__doc__ == expected
        assert tested.foo_bar.__doc__ is None
