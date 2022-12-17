#define PY_SSIZE_T_CLEAN // I'm told this is necessary in python3
#include <Python.h>
//#include "numpy/ndarraytypes.h"
//#include "numpy/npy_3kcompat.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "py3cairo.h"

/////////////////////////////////// MACROS /////////////////////////////////////

#define AssignAndCheck(var, returnval) if((var = returnval) == NULL) return NULL;
#define AssignAndCheckInt(var, returnval) if((var = returnval) == NULL) return -1;
#define AssignAndCheckEr(var, returnval, Er) if((var = returnval) == NULL) Er; return -1;
#define ArgTypeCheck(Obj,type) if (!PyObject_TypeCheck(Obj, type)) return PyErr_Format( \
    PyExc_TypeError, "Expected %s, got %s", type->tp_name, Obj->ob_type->tp_name);

////////////////////////////////// EXTERNS /////////////////////////////////////

extern void threanim_helloworld(char** outstr);

/////////////////////////////////// MEMORY /////////////////////////////////////

typedef struct {
    PyTypeObject *Mobject;
    PyTypeObject *VMobject;
} Manim_Types;

typedef struct {
    Manim_Types type;
} CanimBits;

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// MOBJECT FUNCS //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////




/////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// CAM FUNCS ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

static PyObject *FW_display_vectorized(PyObject *self, PyObject *args){
    /* C version of camera.display_vectorized that can be called from python */
    CanimBits *canim;
    PyObject *vmobject, *context;
    PyArrayObject* points;

    AssignAndCheck(canim, PyModule_GetState(self)); 
    if(!PyArg_ParseTuple(args, "OO", &vmobject, &context)) return NULL;

    ArgTypeCheck(vmobject, canim->type.VMobject);
    //ArgTypeCheck(context, &PycairoContext_Type);

    //AssignAndCheck(points, PyObject_GetAttrString(vmobject, "points"));




    return Py_BuildValue("s", "Type Check completed");
}



/////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// TEST //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

static PyObject* canim_test(PyObject *self, PyObject *args) {
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

static PyObject* threanim_test(PyObject *self, PyObject *args) {
    /*  Example callable function from python that 
        gets hits hello world string from CPP. Note that some care
        is taken to make sure the passed string is of type char* and not
        const char*
    */
    char* str;
    PyObject *pystr;

    if(!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    str = NULL;
    threanim_helloworld(&str);
    pystr = Py_BuildValue("s", str);
    free(str);

    return pystr;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// MEMORY MANAGEMENT ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////



static int canim_modexec(PyObject *m) {
    CanimBits *canim;
    PyObject* manimModule;
    PyObject* placeholder;

    AssignAndCheckInt(canim, PyModule_GetState(m));
    AssignAndCheckInt(manimModule, PyImport_ImportModule("manim"));

    #define AssignTypeAndCheck(place, typeName, placeholder) \
    AssignAndCheckInt(placeholder, PyObject_GetAttrString(manimModule, typeName)); \
    if(placeholder->ob_type != &PyType_Type) { \
        PyErr_Format(PyExc_TypeError, "Expected type in %s for type initialisation, got object", typeName); \
        return -1; \
    } \
    place = (PyTypeObject*) placeholder;

    AssignTypeAndCheck(canim->type.Mobject, "Mobject", placeholder);
    AssignTypeAndCheck(canim->type.Mobject, "VMobject", placeholder);

    #undef AssignTypeAndCheck

    Py_DECREF(manimModule);

    return 0;
}

static PyModuleDef_Slot canim_slots[] = {
    {Py_mod_exec, canim_modexec},
    {0, NULL}
};

static int canim_traverse(PyObject *m, visitproc visit, void *arg) {
    CanimBits *canim;

    AssignAndCheckInt(canim, PyModule_GetState(m));

    Py_VISIT(canim->type.Mobject);
    Py_VISIT(canim->type.VMobject);

    return 0;
}

static int canim_clear(PyObject *m) {
    CanimBits *canim;

    AssignAndCheckInt(canim, PyModule_GetState(m));

    Py_CLEAR(canim->type.Mobject);
    Py_CLEAR(canim->type.VMobject);

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// FINAL SETUP //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

static PyMethodDef CanimMethods[] = {
    /*  Callable functions in the Canim module object must be
        listed here. The format is as follows: function name,
        function pointer, method arguments type, and finally a
        documentation string. The final entry must be a terminator,
        like the \0 at the end of a string.
    */
    {"canim_test", canim_test, METH_VARARGS, "Test Canim"},
    {"threanim_test", threanim_test, METH_VARARGS, "Test Threanim"},
    {"FW_display_vectorized", FW_display_vectorized, METH_VARARGS, "Test typechecking"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef canimmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "canim",
    .m_doc = NULL,
    .m_size = sizeof(CanimBits),
    .m_methods = CanimMethods,
    .m_slots = canim_slots,
    .m_traverse = canim_traverse,
    .m_clear = canim_clear,
};

PyMODINIT_FUNC PyInit_canim(void) {
    PyObject *module;

    module = PyModuleDef_Init(&canimmodule);
    if (module == NULL) return NULL;
    if (import_cairo() < 0) return NULL;

    import_array(); // Numpy prefers to do its check inhouse for some reason
    import_umath();    

    return module;
}

