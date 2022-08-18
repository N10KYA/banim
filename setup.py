from distutils.core import setup, Extension

def main():
    setup(name='canim',
          version='1.0',
          ext_modules=[Extension('canim', ['canim.c'])])