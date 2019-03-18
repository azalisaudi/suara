import sys
#In Ubuntu 14.04.2, the pocketsphinx module shows error in first
import and will work for the second import. The following code is
a temporary fix to handle that issue
try:
import pocketsphinx
except:
import pocketsphinx