#use setup tools, not distutils which is depreciated, but used about the same
from subprocess import call
from setuptools import setup, Extension


# not sure if this structure is necessary but I think the setup func needs to be run with a python argument to 'develop'
# also, use develop as opposed to install so that it installs the package in the current environment, not the global one

#care to change include directory, and/or adjust compile arguments for other distributions
def main():
    call(["clang", "-c", "threanim.cpp", "--include-directory", "/usr/include/cairo/", "-o", "threanim.o"])
    ext=Extension('canim', ['canimModule.c','caironim.c'], extra_objects=['threanim.o'])
    setup(name='canim',
          version='test0.0.1',
          ext_modules=[ext])

if __name__ == '__main__':
    main()