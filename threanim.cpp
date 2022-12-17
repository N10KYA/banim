//#define PY_SSIZE_T_CLEAN
//#include <Python.h>
#include <stdlib.h>
#include <thread>


// This function is declared as extern "C" to avoid name mangling
extern "C" void threanim_helloworld(char** outstr)
{
    
    /**/
    // The message to be displayed
    const char* theMessage = "Hello World From Threanim!";
    int length=0;
    
    // Determine the length of the string
    while(theMessage[length]!='\0'){length++;}
    
    // Allocate memory to hold the string
    *outstr = (char*)malloc((length+1) * sizeof(char));

    // Copy the string into the new memory location
    for(int i = 0; i < length; i++) {(*outstr)[i] = theMessage[i];}
    // Add the null terminator to the end of the string
    (*outstr)[length+1]= '\0';
}