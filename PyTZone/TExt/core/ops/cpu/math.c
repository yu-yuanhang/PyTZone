#include "math.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

float ca_max(float a, float b)
{
    float m;
    m = 0;
    if(a > b){
        m = a;
    }else
    {
        m = b;
    }
    return m;
}


double ca_pow(double a, int n)
{
    if(n<0) return 1/ca_pow(a,-n);
    double res = 1.0;
    while(n)
    {
        if(n&1) res *= a;
        a *= a;
        n >>= 1;
    }
    return res;
}


double ca_eee(double x)
{
    if(x>1e-3)
    {
        double ee = ca_eee(x/2);
        return ee*ee;
    }
    else
        return 1 + x + x*x/2 + ca_pow(x,3)/6 + ca_pow(x,4)/24 + ca_pow(x,5)/120;
}


double ca_exp(double x)
{
    if(x<0) return 1/ca_exp(-x);
    int n = (int)x;
    x -= n;
    double e1 = ca_pow(e,n);
    double e2 = ca_eee(x);
    return e1*e2;
}


//  float ca_rand()
//  {
//      uint16_t digest[20];
//      int sumrand = 0;
//      TEE_GenerateRandom(digest,sizeof(digest));

//      for(int i=0; i<20; i++){
//            unsigned int digest_int = digest[i];
//            sumrand += digest_int;
//      }
//      float rand_num = (float) (sumrand % 10000) / 10000;

//      //char char0[30];
//      //ftoa(rand_num, char0, 5);
//      //IMSG("rand_num=%s\n",char0);

//      return rand_num;
//  }

/*
#include <time.h>
#include <stdlib.h>

float ca_rand()
{
    //srand(time(0));   // Initialization, should only be called once.
    float r = (float)rand()/RAND_MAX;      // Returns a pseudo-random integer between 0 and RAND_MAX.
    return r;
}
*/


int ca_floor(double x)
{
    return (int) x - (x< 0);
}

double ca_sqrt(double x)
{
    if(x>100) return 10.0*ca_sqrt(x/100);
    double t = x/8 + 0.5 + 2*x/(4+x);
    int c = 10;
    while(c--)
    {
        t = (t+x/t)/2;
    }
    return t;
}


double F1(double x)
{
    return 1/x;
}

double F2(double x)
{
    return 1/ca_sqrt(1-x*x);
}

double simpson(double a, double b,int flag)
{
    double c = a + (b-a)/2;
    if(flag==1)
        return (F1(a)+4*F1(c)+F1(b))*(b-a)/6;
    if(flag==2)
        return (F2(a)+4*F2(c)+F2(b))*(b-a)/6;
}

double asr(double a, double b, double eps, double A,int flag)
{
    double c = a + (b-a)/2;
    double L = simpson(a, c,flag), R = simpson(c, b,flag);
    if(fabs(L+R-A) <= 15*eps) return L+R+(L+R-A)/15.0;
    return asr(a, c, eps/2, L,flag) + asr(c, b, eps/2, R,flag);
}

double asr0(double a, double b, double eps,int flag)
{
    return asr(a, b, eps, simpson(a, b,flag),flag);
}

double ca_ln(double x)
{
    return asr0(1,x,1e-8,1);
}

double ca_log(double a,double N)
{
    return ca_ln(N)/ca_ln(a);
}

double ca_sin(double x)
{
    double fl = 1;
    if(x>2*PI || x<-2*PI) x -= (int)(x/(2*PI))*2*PI;
    if(x>PI) x -= 2*PI;
    if(x<-PI) x += 2*PI;
    if(x>PI/2)
    {
        x -= PI;
        fl *= -1;
    }
    if(x<-PI/2)
    {
        x += PI;
        fl *= -1;
    }
    if(x>PI/4) return ca_cos(PI/2-x);
    else return fl*(x - ca_pow(x,3)/6 + ca_pow(x,5)/120 - ca_pow(x,7)/5040 + ca_pow(x,9)/362880);
}

double ca_cos(double x)
{
    double fl = 1;
    if(x>2*PI || x<-2*PI) x -= (int)(x/(2*PI))*2*PI;
    if(x>PI) x -= 2*PI;
    if(x<-PI) x += 2*PI;
    if(x>PI/2)
    {
        x -= PI;
        fl *= -1;
    }
    if(x<-PI/2)
    {
        x += PI;
        fl *= -1;
    }
    if(x>PI/4) return ca_sin(PI/2-x);
    else return fl*(1 - ca_pow(x,2)/2 + ca_pow(x,4)/24 - ca_pow(x,6)/720 + ca_pow(x,8)/40320);
}

double ca_tan(double x)
{
    return ca_sin(x)/ca_cos(x);
}

// reverses a string 'str' of length 'len'
void reverse(char *str, int len)
{
    int i=0, j=len-1, temp;
    while (i<j)
    {
        temp = str[i];
        str[i] = str[j];
        str[j] = temp;
        i++; j--;
    }
}

// Converts a given integer x to string str[].  d is the number
// of digits required in output. If d is more than the number
// of digits in x, then 0s are added at the beginning.
int intToStr(int x, char str[], int d)
{
    int i = 0;
    while (x)
    {
        str[i++] = (x%10) + '0';
        x = x/10;
    }

    // If number of digits required is more, then
    // add 0s at the beginning
    while (i < d)
        str[i++] = '0';

    reverse(str, i);
    str[i] = '\0';
    return i;
}

// Converts a floating point number to string.
void ftoa(float n_orig, char *res, int afterpoint)
{
    // handle negative
    float n = 0.0f;
    if(n_orig < 0){
        n = n_orig * -1;
    }else{
        n = n_orig;
    }

    // Extract integer part
    int ipart = (int)n;

    // Extract floating part
    float fpart = n - (float)ipart;

    // convert integer part to string
    int i = intToStr(ipart, res, 0);

    // check for display option after point
    if (afterpoint != 0)
    {
        res[i] = '.';  // add dot

        // Get the value of fraction part upto given no.
        // of points after dot. The third parameter is needed
        // to handle cases like 233.007
        fpart = fpart * ca_pow(10, afterpoint);

        intToStr((int)fpart, res + i + 1, afterpoint);
    }
}


void bubble_sort_top(float *arr, int len) {
    int i, j;
    float temp;

    // sort
    for (i = 0; i < len - 1; i++){
        for (j = 0; j < len - 1 - i; j++){
            if (arr[j] < arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
