project('manim_renderer', 'c', 'cpp',
  version : '0.1.0',
  default_options : [
    'warning_level=3',
    'buildtype=release', # this is needed so python debug files aren't searched for
  ]
)

pymod = import('python')
python = pymod.find_installation(pure: false)

pyver = python.language_version()
if pyver.version_compare('< 3.7')
  error('Requires Python >= 3.7')
endif

cc = meson.get_compiler('c')

incdir_numpy = run_command(python,
  [
    '-c',
    'import os; import numpy; print(os.path.relpath(numpy.get_include()))'
  ],
  check: true
).stdout().strip()
inc_np = include_directories(incdir_numpy)

incdir_pycairo = run_command(python,
  [
    '-c',
    'import os; import cairo; print(os.path.relpath(cairo.get_include()))'
  ],
  check: true
).stdout().strip()
inc_pycairo = include_directories(incdir_pycairo)

cairo_dep = dependency('cairo', required: true, 
      version: '>= 1.16.0', fallback: ['cairo', 'cairo_dep'])

python_dep = python.dependency()

python.extension_module(
    'canim',
    'canimModule.c',
    'threanim.cpp',
    dependencies: [cairo_dep],
    include_directories: [incdir_numpy, inc_pycairo],
    install: true,
)
