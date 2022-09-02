#define PY_SSIZE_T_CLEAN // I'm told this is necessary in python3
#include <Python.h>

extern char *danim_helloworld(void);



static PyObject *canim_test(PyObject *self, PyObject *args) {
    /*  Example callable function from python.
        PyArg_ParseTuple should have more arguments if the
        method takes arguments, see later example.
    */
    if(!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return Py_BuildValue("s", "Hello From Canim");
    // Py_BuildValue converts a C value to a Python object
}

static PyObject *danim_test(PyObject *self, PyObject *args) {
    /*  Example callable function from python that 
        gets hits hello world string from CPP. Note that some care
        is taken to make sure the passed string is of type char* and not
        const char*
    */
    char *str;

    if(!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    str = danim_helloworld();

    return Py_BuildValue("s", str);
}


static PyMethodDef CanimMethods[] = {
    /*  Callable functions in the Canim module object must be
        listed here. The format is as follows: function name,
        function pointer, method arguments type, and finally a
        documentation string. The final entry must be a terminator,
        like the \0 at the end of a string.
    */
    {"canim_test", canim_test, METH_VARARGS, "Test Canim"},
    {"danim_test", danim_test, METH_VARARGS, "Test Danim"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef canimmodule = {
    PyModuleDef_HEAD_INIT,
    "canim",
    NULL,
    -1, /*  -1 seems to mean that the module keeps no persistent memory
            Change to 1 if this library begins storing its own variables
            between python function calls (I think?). Revise this comment.
    */
    CanimMethods
};

PyMODINIT_FUNC PyInit_canim(void) {
    return PyModule_Create(&canimmodule);
}

