import unittest
from core import pattern

class TestCore(unittest.TestCase):
    
    def test_init_antenna_array(self):
        # Test empty pattern initializer
        pat = pattern()
        self.assertHasAttr(pat, 'data_array')

    def assertHasAttr(self, obj, attribute_name, msg=None):
        if not hasattr(obj, attribute_name):
            self.fail(msg)
    
    def assertDoesNotHasAttr(self, obj, attribute_name, msg=None):
        if hasattr(obj, attribute_name):
            self.fail(msg)

if __name__ == '__main__':
    unittest.main()