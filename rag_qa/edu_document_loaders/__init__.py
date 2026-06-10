import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from edu_docloader import *
from edu_pptloader import *
from edu_imgloader import *
from edu_pdfloader import *
