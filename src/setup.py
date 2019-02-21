import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ddestimator = Extension("ddestimator", ["ddestimator.pyx"],
                            include_dirs=[np.get_include()],
                            libraries=["numpy","imutils","dlib","cv2","time","math","re","pandas","scipy","lmfit"]
                        )
setup(cmdclass={'build_ext': build_ext},
      ext_modules=[ddestimator]
     )