"""
This is an example that shows how I program.
The aim of this program is to wrap the print function in a class.
"""

class Print:
	def __init__(self, functor=print):
		""" Initialise the Print object
			Args:
				functor:
				function used to write or to print a string, by default the print function is used.
			Returns:
			none.
		"""
		self.functor = functor
		
	def original(self, *args, **kwargs):
		""" Use the functor to print/write the original text (string)
			args:
				text:
				the text to print/write

				kwargs:
				the additional options
			Returns:
			none:			
		"""
		self.functor(*args, **kwargs)
	
	def lower(self, *args, **kwargs):
		""" Use the functor to print/write a lower case text (string)
			Args:
				args:
				the text to print/write

				kwargs:
				the additional options
			Returns:
			none:	
		"""
		sep = " "
		try:
			sep = str(kwargs['sep'])
		except:
			pass

		self.functor((sep.join(map(str, args))).lower(), **kwargs)
		
	def upper(self, *args, **kwargs):
		""" Use the functor to print/write a upper case text (string)
			Args:
				args:
				the text to print/write

				kwargs:
				the additional options
			Returns:
			none:	
		"""
		sep = " "
		try:
			sep = str(kwargs['sep'])
		except:
			pass

		self.functor((sep.join(map(str, args))).upper(), **kwargs)
		
if __name__ == "__main__":
	# The previous if statement allows to execute the code only if we are using it as the main program.
	
	# Tests
	my_print = Print()
	
	# Print.original()
	my_print.original("Original", end="\n\r")
	
	# Print.lower()
	my_print.lower("Lower", end="\n\r")
	
	# Print.upper()
	my_print.upper("Upper", "lower?", end="\n\r", sep='\t')