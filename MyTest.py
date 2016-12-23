import unittest
import numpy as np
import Simplex

class Test:
	def __init__(self):
		pass

	def testSimplex(self, list):
		return Simplex.solve(*list)	

class TestSimplex(unittest.TestCase):
	def setUp(self):
		self.list = Test()

	# example of sec 3 chap 6, primal simplex, z = 31
	def testPrimalSimplex(self):
		title = "Testing primal simplex"
		A = np.matrix([1.0, 2.0, 0, -1.0, -1.0, 1.0])
		A = A.reshape((2, 3))
		b = np.matrix([1.0, 3.0, 5.0])
		c = np.matrix([4.0, 3.0])
		self.assertAlmostEqual(31.0, self.list.testSimplex([A, b, c]))

	# problem 2.1, primal simplex, z = 17
	def testSimplexTwo(self):		
		title = "Testing primal simplex"
		A = np.matrix([2.0, 1.0, 1, 3.0, 1.0, 1.0, 3.0, 2.0])
		A = A.reshape((4, 2))
		b = np.matrix([5.0, 3.0])
		c = np.matrix([6.0, 8.0, 5.0, 9.0])
		self.assertAlmostEqual(17.0, self.list.testSimplex([A, b, c]))

	# problem 2.4, dual simplex, z = -3
	def testDual(self):
		title = "Testing dual simplex"
		A = np.matrix([2.0, 2.0, -5.0, -1.0, 1.0, 2.0])
		A = A.reshape((3, 2))
		b = np.matrix([-5.0, 4.0])
		c = np.matrix([-1.0, -3.0, -1.0])
		self.assertAlmostEqual(-3.0, self.list.testSimplex([A, b, c]))

	# problem 2.3, 2-phase, z = -3
	def testTwoPhase(self):
		title = "Testing 2-phase simplex"
		A = np.matrix([-1.0, 2.0, -1.0, -1.0, -1.0, 1.0])
		A = A.reshape((3, 2))
		b = np.matrix([-2.0, 1.0])
		c = np.matrix([2.0, -6.0, 0.0])
		self.assertAlmostEqual(-3.0, self.list.testSimplex([A, b, c]))

	# example of sec 7 chap 5, 2-phase, unbounded
	def testTwoPhaseTwo(self):
		title = "Testing 2-phase simplex"
		A = np.matrix([-2, -2, -1, -1, 4, 3])
		A = A.reshape((2, 3))
		b = np.matrix([4, -8, -7])
		c = np.matrix([-1, 4])
		self.assertAlmostEqual("unbounded", self.list.testSimplex([A, b, c]))	

	# using strong dual theorem to test the optimality. same dictionary as test 1
	def testOptimality(self):		
		title = "Testing primal simplex"
		A = np.matrix([1.0, 2.0, 0, -1.0, -1.0, 1.0])
		A = A.reshape((2, 3))
		b = np.matrix([1.0, 3.0, 5.0])
		c = np.matrix([4.0, 3.0])
		self.assertAlmostEqual(0, self.list.testSimplex([A, b, c]) + self.list.testSimplex([-np.transpose(A), -c, -b]))

	# trivial case, where the dictionary is already optimal. dictionary modified from test 1
	def testTrivial(self):		
		title = "Testing primal simplex"
		A = np.matrix([1.0, 2.0, 0, -1.0, -1.0, 1.0])
		A = A.reshape((2, 3))
		b = np.matrix([1.0, 3.0, 5.0])
		c = np.matrix([-4.0, -3.0])
		self.assertAlmostEqual("optimal", self.list.testSimplex([A, b, c]))

def suite():
	suite = unittest.TestSuite()
	suite.addTest(unittest.makeSuite(TestSimplex))
	return suite

if __name__ == "__main__":
	unittest.TextTestRunner(verbosity=2).run(suite())	
