#include <cairo.h>
#include <thread>


extern "C" char *danim_helloworld(void)
{
    const char* theMessage = "Hello World From Danim!";
    int length=0;
    while(theMessage[length]!='\0'){length++;}

    char str[length + 1];
    for(int i = 0; i < length; i++) {str[i] = theMessage[i];}
    str[length+1]= '\0';

    return str;
}

