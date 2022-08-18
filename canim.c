#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *canim_test(PyObject *self, PyObject *args) {
    if(!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return Py_BuildValue("s", "Hello` From Canim");
}

static PyMethodDef CanimTestMethod[] = {
    {"test", canim_test, METH_VARARGS, "Test Canim"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef canimmodule = {
    PyModuleDef_HEAD_INIT,
    "canim",
    NULL,
    -1,
    CanimTestMethod
};

PyMODINIT_FUNC PyInit_canim(void) {
    return PyModule_Create(&canimmodule);
}