from math import *
from sympy.abc import x, s, t, j, a, w, v, n, m, k, y, z, e, r, u
from sympy import sqrt, sin, cos, tan, log, series, integrate, evalf, re, im, oo, solve, Abs, inverse_laplace_transform, simplify, inverse_fourier_transform, solve_linear_system_LU
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from fractions import Fraction
import os, errno
from time import gmtime, strftime, time
import sys
import webbrowser as wb
import threading, queue
import sympy.assumptions.handlers.calculus

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

class functions:
    '''
        This class simply contains static methods.
    '''

    def focusedWidget(self, btn, widgets):
        widget = None

        def setFocus(self):
            widget.focus = True

        for i in widgets:
            if i.focus:
                widget = i
                if btn.text == 'Clear':
                    i.text = ''
                else:
                    i.text += btn.text
                btn.bind(on_release=setFocus)
    @staticmethod
    def restart_program():
        """Restarts the current program.
        Note: this function does not return. Any cleanup action (like
        saving data) must be done before calling this function."""
        python = sys.executable
        os.execl(python, python, * sys.argv)

    @staticmethod
    def _quadraticroot_(i, a, b, c):
        if i == 1:
            return ((-b + sqrt(b * b - 4 * a * c)) / (2 * a))
        else:
            return ((-b - sqrt(b * b - 4 * a * c)) / (2 * a))

    @staticmethod
    def linear_and_non_equations(L='[]', var='[]', a=0, b=0, c=0, d=0, e=0, f=0, kind=0):
        if kind == 1:
            # Linear equations
            d = solve_linear_system_LU(sp.Matrix(L), parse_expr(var))
            for i in d:
                d[i] = d[i].evalf()
            return str(d)
        elif kind == 2:
            # Quadratic equations
            try:
                L = solve(a * x ** 2 + b * x + c - d, x)
                flag = 0
                if len(L) == 1:
                    # The equation has a repeated root
                    a = str(L[0])
                    for i in a:
                        if i.isalpha():
                            L[0] = L[0].evalf()
                            flag = 1
                    if flag == 0:
                        return 'The repeated root is: {:}'.format(L[0])
                    else:
                        return 'The repeated root is: {:s}'.format(functions.float_answer_analyzer(L[0]))
                elif len(L) != 1 and not functions.complex_checker(L[0]):
                    for i in L:
                        a = str(i)
                        for j in a:
                            if j.isalpha() and j != 'j':
                                L[0] = L[0].evalf();
                                L[1] = L[1].evalf()
                                flag = 1
                    # Equation with two distinct roots
                    if flag == 0:
                        return 'The roots are: {:} and {:}'.format(re(L[0]), re(L[1]))
                    else:
                        return 'The roots are: {:s} and {:s}'.format(functions.float_answer_analyzer(re(L[0])),
                                                                       functions.float_answer_analyzer(re(L[1])))
                else:
                    for i in L:
                        a = str(i)
                        for j in a:
                            if j.isalpha() and j != 'j':
                                L[0] = L[0].evalf();
                                L[1] = L[1].evalf()
                                flag = 1
                    # Equation with complex roots
                    if flag == 0:
                        return 'The roots are: {:}+{:}j and  {:}-{:}j'.format(re(L[0]), im(L[0]), re(L[1]), im(L[1]))
                    else:
                        return 'The roots are: {:s}+{:s}j and  {:s}-{:s}j'.format(
                            functions.float_answer_analyzer(re(L[0])), functions.float_answer_analyzer(Abs(im(L[0]))),
                            functions.float_answer_analyzer(re(L[1])), functions.float_answer_analyzer(Abs(im(L[1]))))
            except:
                return 'Math error!'
        elif kind == 3:
            # Cubic Equations
            try:
                l = solve(a * x ** 3 + b * x ** 2 + c * x + d - e, x)
                L = []
                for i in l:
                    a = i.evalf()
                    L.append(a)
                ans = []
                for i in L:
                    b = str(i)
                    for j in b:
                        if j == 'I':
                            ans.append('{:s}{:1s}{:s}j'.format(functions.float_answer_analyzer(re(i)),
                                                               '+' if im(i) > 0 else '-',
                                                               functions.float_answer_analyzer(Abs(im(i)))))
                            break

                    else:
                        ans.append('{:s}'.format(functions.float_answer_analyzer(i)))
                return 'The roots are: {:s},\n{:s}\nand {:s}'.format(ans[0], ans[1], ans[2])
            except:
                return 'Math error!'
        elif kind == 4:
            # Quartic equations
            try:
                l = solve(a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e - f, x)
                L = []
                for i in l:
                    a = i.evalf()
                    L.append(a)
                ans = []
                for i in L:
                    b = str(i)
                    for j in b:
                        if j == 'I':
                            ans.append('{:s}{:1s}{:s}j'.format(functions.float_answer_analyzer(re(i)),
                                                               '+' if im(i) > 0 else '-',
                                                               functions.float_answer_analyzer(Abs(im(i)))))
                            break

                    else:
                        ans.append('{:s}'.format(functions.float_answer_analyzer(i)))
                return 'The roots are: {:s},\n{:s},\n{:s}\nand {:s}'.format(ans[0], ans[1], ans[2], ans[3])
            except:
                return 'Math error!'

    @staticmethod
    def plots(F = [], kind=0):
        try:
            if kind == 1:
                # Plotting One cartesian function
                def callback():
                    p = sp.plotting.plot(parse_expr(functions.tipa_parse_str(F[0])),
                                         parse_expr(functions.tipa_parse_str(F[1])))
                thread = threading.Thread(target=callback)
                thread.start()
            elif kind == 2:
                # Plotting two cartesian functions
                def callback():
                    p = sp.plotting.plot((parse_expr(functions.tipa_parse_str(F[0])),
                                      parse_expr(functions.tipa_parse_str(F[1]))),
                                     (parse_expr(functions.tipa_parse_str(F[2])),
                                      parse_expr(functions.tipa_parse_str(F[3]))))

                thread = threading.Thread(target=callback)
                thread.start()
            elif kind == 3:
                # Plotting three cartesian functions
                def callback():
                    p = sp.plotting.plot((parse_expr(functions.tipa_parse_str(F[0])),
                                      parse_expr(functions.tipa_parse_str(F[1]))),
                                     (parse_expr(functions.tipa_parse_str(F[2])),
                                      parse_expr(functions.tipa_parse_str(F[3]))),
                                     (parse_expr(functions.tipa_parse_str(F[4])),
                                      parse_expr(functions.tipa_parse_str(F[5]))))

                thread = threading.Thread(target=callback)
                thread.start()
            elif kind == 4:
                # 2D Plot Given in parametric
                def callback():
                    p = sp.plotting.plot_parametric(parse_expr(functions.tipa_parse_str(F[0])),
                                                parse_expr(functions.tipa_parse_str(F[1])),
                                                parse_expr(functions.tipa_parse_str(F[2])))

                thread = threading.Thread(target=callback)
                thread.start()
            elif kind == 5:
                # 3D plot given in parametric
                def callback():
                    p = sp.plotting.plot_parametric(parse_expr(functions.tipa_parse_str(F[0])),
                                                parse_expr(functions.tipa_parse_str(F[1])),
                                                parse_expr(functions.tipa_parse_str(F[2])),
                                                parse_expr(functions.tipa_parse_str(F[3])))

                thread = threading.Thread(target=callback)
                thread.start()
            elif kind == 6:
                # 3D Surface plot using a cartesian equation
                def callback():
                    p = sp.plotting.plot3d(parse_expr(functions.tipa_parse_str(F[0])),
                                       parse_expr(functions.tipa_parse_str(F[1])),
                                       parse_expr(functions.tipa_parse_str(F[2])))

                thread = threading.Thread(target=callback)
                thread.start()
            elif kind == 7:
                # 3D surface plot using parametric equations
                def callback():
                    p = sp.plotting.plot_parametric(parse_expr(functions.tipa_parse_str(F[0])),
                                                parse_expr(functions.tipa_parse_str(F[1])),
                                                parse_expr(functions.tipa_parse_str(F[2])),
                                                parse_expr(functions.tipa_parse_str(F[3])),
                                                parse_expr(functions.tipa_parse_str(F[4])))

                thread = threading.Thread(target=callback)
                thread.start()
            elif kind == 8:
                # Plots of implicit functions
                def callback():
                    p = sp.plotting.plot_implicit(parse_expr(functions.tipa_parse_str(F[0])),
                                              parse_expr(functions.tipa_parse_str(F[1])),
                                              parse_expr(functions.tipa_parse_str(F[2])))

                thread = threading.Thread(target=callback)
                thread.start()
            elif kind == 9:
                # Plots of Inequalities
                def callback():
                    p = sp.plotting.plot_implicit(parse_expr(functions.tipa_parse_str(F[0])))

                thread = threading.Thread(target=callback)
                thread.start()
        except:
            pass
   
    @staticmethod
    def Webpage(a = False):
        if a:
            wb.open_new_tab('http://tipan-coders.simplesite.com/440139914')
        else:
            wb.open_new_tab('http://tipan-coders.simplesite.com/440139921')

    def character_checker(x):
        char = '+-*/'
        for i in range(len(char)-1):
            if char[i] == x:
                return True
            elif i == (len(x)-1):
                return False

    @staticmethod
    def tipa_parse_str(a, result='', mode=True):
        """ This function parses a text input in such a way that users can input mathematical expressions in a more friendly way
            it takes on only one argument which is a string, and returns a string as well.
            """
        a = a.replace('Ans', result)
        a = a.replace('I', 'j')
        a = a.replace('\u2044', '/')
        a = a.replace('\u00b2', '**2')
        a = a.replace('\u00b3', '**3')
        a = a.replace('^', '**')
        a = a.replace('√', 'sqrt')
        a = a.replace('\u221E', 'oo')
        a = a.replace('E', '*10**')
        a = a.replace('Ans', result)
        a = a.replace('Log', 'log10')
        a = a.replace('ln', 'log')
        a = a.replace('exp', 'exp')
        a = a.replace('\u00D7', '*')
        a = a.replace('\u005E', '**')
        a = list(a)
        for i in range(len(a) - 1):
            if a[i].isdigit() and (a[i + 1].isalpha() or a[i + 1] == '√') and a[i + 1] != 'j':
                a[i] = a[i] + '*'
            if a[i].isdigit() and a[i + 1] == 'x':
                a[i + 1] = '*' + a[i + 1]
            if a[i] == 'x' and a[i + 1].isalpha() and a[i + 1] != 'p':
                a[i + 1] = '*' + a[i + 1]
            if a[i] == ')' and a[i + 1].isalpha():
                a[i + 1] = '*' + a[i + 1]
            if (a[i].isalpha() or a[i] == '√') and (a[i + 1].isdigit() or a[i + 1] == '.'):
                a[i] = a[i] + '('
                b = a[i + 1:]
                j = 0
                while True:
                    if b[j] == '+' or b[j] == '-' or b[j] == '*' or b[j] == '/':
                        b[j] = ')' + b[j]
                        break
                    elif j == len(b) - 1:
                        b[j] = b[j] + ')'
                        break
                    j += 1
                a = a[:i + 1] + b[:]
        a = ''.join(a)
        if not mode:
            a = a.replace('sin', 'Sin')
            a = a.replace('cos', 'Cos')
            a = a.replace('tan', 'Tan')
            a = a.replace('asin', 'aSin')
            a = a.replace('acos', 'aCos')
            a = a.replace('atan', 'aTan')
            a = a.replace('arg', 'Arg')

        return a

    @staticmethod
    def simpsons_rule(f='', a='', b=''):
        """ This performs definite integral by the method of approximation based on simpson's rule, it takes on three arguments, f, a and b
            f: is the function to be operated upon which is normally in string form
            a: lower limit of integration
            c: upper limit of integration
            It returns a floating point value"""
        if a == b == '':
            return simplify(integrate(f, x, conds='none'))
        else:
            try:
                a = eval(a)
                b = eval(b)
            except: pass
            n = 100;
            h = (b - a) / n
            x1 = [(a + i * h) for i in range(n + 1)]
            f = parse_expr(f)
            y = [f.subs(x, t).evalf() for t in x1]
            E = [y[t] for t in range(len(y)) if (t != 0 and t % 2 == 0 and t != n)]
            R = [y[t] for t in range(len(y)) if (t != 0 and t % 2 != 0 and t != n)]
            sum1 = y[0] + y[n]
            sum2 = sum(E)
            sum3 = sum(R)
            A = h * (sum1 + 2 * sum2 + 4 * sum3) / 3
            return A

    @staticmethod
    def math_display(a):
        """ This function modifies a string so that it is output in more mathematical format,
        it takes on a single argument in string format and returns a string
        For Example:
        a = '3*x**2*sin(x**3)'
        b = math_display(a)
        b
        '3x\u00b2sin(x\u00b3)'
        """
        a = list(a)

        def superscript(t):
            a = {0: '\u2070', 1: '\u00B9', 2: '\u00b2', 3: '\u00b3', 4: '\u2074', 5: '\u2075', 6: '\u2076',
                 7: '\u2077', 8: '\u2078', 9: '\u2079'}
            if t.isdigit():
                return a[int(t)]

        for i in range(len(a) - 3):
            if a[i] == a[i + 1] == '*' and a[i + 2].isdigit() and a[i + 3].isdigit():
                a[i] = superscript(a[i + 2]) + superscript(a[i + 3]);
                a[i + 1] = a[i + 2] = a[i + 3] = ''
            if a[i] == a[i + 1] == '*' and (not a[i + 2].isdigit()):
                a[i] = '^';
                a[i + 1] = ''
            if a[i] == a[i + 1] == '*' and a[i + 2].isdigit() and (not a[i + 3].isdigit()):
                a[i] = superscript(a[i + 2]);
                a[i + 1] = a[i + 2] = ''
            if a[i] == '*' and a[i + 1] != '*':
                a[i] = ''
            if a[i] == 'l' and a[i + 1] == 'o' and a[i + 2] == 'g':
                a[i] = 'ln';
                a[i + 1] = a[i + 2] = ''
            if a[i] == 's' and a[i + 1] == 'q' and a[i + 2] == 'r' and a[i + 3] == 't':
                a[i] = '√';
                a[i + 1] = a[i + 2] = a[i + 3] = ''
        a = ''.join(a)
        return a

    @staticmethod
    def decimal_to_fraction(y):
        y = eval(str(y))
        if not functions.complex_checker(str(y)):
            try:
                a = y.numerator
                number = y
            except:
                number = y
                number = number.as_integer_ratio()
                number = Fraction(number[0]/number[1]).limit_denominator()
            return str(number).replace('/', '\u2044')
        else:
            n = y
            a = re(n)
            b = im(n)
            try:
                a_num = a.numerator
                b_num = b.numerator
                number = eval(y)
                print('I am in the 1st shit')
                return str(number)
            except:
                print('I am in the 2nd shit')
                a = float(a)
                b = float(b)
                a = a.as_integer_ratio()
                b = b.as_integer_ratio()
                a = Fraction(a[0]/a[1]).limit_denominator()
                b = Fraction(b[0]/b[1]).limit_denominator()
                i = str(abs(b))
                i = i.replace('/', 'j\u2044')
                if not functions.complex_checker(i):
                    i = i + 'j'
                print('imaginary = ' + '{:s}{:s}{:s}'.format(str(a), '+' if b>0 else '-', i))
                return '{:s}{:s}{:s}'.format(str(a), '+' if b>0 else '-', i)

    @staticmethod
    def apostrophe(x):
        try:
            a = x.count('\u2032')
            b = x.count('\u2033')
            c = x.count('\u2034')
            if a == b == c == 0:
                y = '\u2032'
            elif a == 1 and b == c == 0:
                y = '\u2033'
            elif a == b == 1 and c == 0:
                y = '\u2034'
            return y
        except:
            functions.speaking('Please you can not alter the standard format, the last value is microseconds.')
            return ''
        
    @staticmethod
    def TaylorSeries(f, b, a = 0):
        ans = functions.Maclaurins_series(f, b)
        ans = functions.math_display(str(ans))
        return ans
    
    @staticmethod
    def complex_checker(n):
        try:
            n = eval(n)
            a = re(n)
            b = im(n)
            if b != 0:
                return True
            else:
                return False
        except:
            return False

    @staticmethod
    def serialisation():
        filename = "C:/Tipan-App/data/bin.txt"
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        try:
            with open(filename, 'r') as f:
                activation_detail = [i.strip() for i in f]
                a_detail = activation_detail[2].split(' ')
                validation_key = activation_detail[3].split(' ')
                key = a_detail[1]
                if len(key) == 20:
                    raise
                key = key[4:-4]
                v_key = validation_key[1].upper()
                v_key = v_key[-1] + v_key[1:-1] + v_key[0]
            if len(key) == 16 and key[-1].isdigit() and key == v_key and key != 'Free':
                return [1]
            else:
                return [2]
        except:
            try:
                if functions.trial_validation():
                    return 1
                else:
                    return 2
            except:
                return 3
    @staticmethod
    def trial_validation():
        with open("C:/Tipan-App/data/bin.txt", 'r') as f:
            a = [i.strip() for i in f]
            b = a[4].split(' ')
            start_time = eval(b[1])
            current_time = time()
            if current_time - start_time < 432000:
                return True
            else:
                return False

    @staticmethod
    def float_analyzer(a):
        a = float(a)
        if a.is_integer():
            a = int(a)
            return str(a)
        else:
            number = a.as_integer_ratio()
            den = str(number[1])
            if len(den) <= 3:
                number = Fraction(number[0]/number[1]).limit_denominator()
                return str(number)
            else:
                return str(a)

            
    @staticmethod
    def float_answer_analyzer(a):
        a = float(a)
        if a.is_integer():
            a = int(a)
            return str(a)
        else:
            number = a.as_integer_ratio()
            den = str(number[1])
            if len(den) <= 3:
                number = Fraction(number[0]/number[1]).limit_denominator()
                return str(number)
            else:
                return '{:.4f}'.format(a)

    @staticmethod
    def is_numeric(a):
        try:
            float(eval(a))
            return True
        except:
            return False

        
    @staticmethod
    def Maclaurins_series(f, b, a = 0):
        f = parse_expr(f)
        L = []
        for i in range(a, b + 1):
            if i == 0:
                try:
                    ans = f.subs(x, 0).evalf()
                    ans = functions.decimal_to_fraction(ans)
                except:
                    ans = ''
            else:
                try:
                    c = sp.diff(f, x, i, evaluate=False)
                    c = c.subs(x, 0).evalf()
                    c = c / fact(i)
                    c = functions.decimal_to_fraction(c)
                    c = '' if c == '1' else '(' + c + ')'
                    x_value = 'x' if i == 1 else 'x**{:d}'.format(i)
                    ans = c + x_value
                except:
                    ans = ''
            L.append(ans)
        ans = functions.math_display('+'.join(L))
        return ans

    @staticmethod
    def Fourier_Transform(f, a, b):
        f = 'exp(-I*w*x) * ' + '(' + f + ')'
        f = functions.tipa_parse_str(f)
        f = parse_expr(f)
        ans = simplify(integrate(f, (x, a, b), conds='none'))
        return ans

    @staticmethod
    def Laplace_Transform(f):
        from sympy.functions.elementary import trigonometric as tg
        f = 'exp(-s*x) * ' + '(' + f + ')'
        f = functions.tipa_parse_str(f)
        print(f)
        f = parse_expr(f)
        print(f)
        ans = simplify(integrate(f, (x, 0, oo), conds='none'))
        return ans

    @staticmethod
    def special_functions(v=0, u = 0, w=0, kind = 0):
        q = queue.Queue()
        def callback(q, v=0, u = 0, w=0, kind = 0):
            ans = ''
            if kind == 1:
                ans = str(sp.gamma(v))
            elif kind == 2:
                ans = str(sp.gamma(u) * sp.gamma(v) / sp.gamma(u + v))
            elif kind == 3:
                ans = str(functions.Legendre_Polynomials(v))
            elif kind == 4:
                ans = str(sp.assoc_legendre(v, u, x))
            elif kind == 5:
                ans = str(functions.bessel_function_1st(v))
            elif kind == 6:
                ans = str(sp.jacobi(u, v, w, x))
            elif kind == 7:
                ans = str(sp.jacobi_normalized(u, v, w, x))
            elif kind == 8:
                ans = str(sp.gegenbauer(u, v, x))
            elif kind == 9:
                # 1st kind
                ans = str(sp.chebyshevt(u, x))
            elif kind == 10:
                ans = str(sp.chebyshevt_root(u, v))
            elif kind == 11:
                # 2nd kind
                ans = str(sp.chebyshevu(u, x))
            elif kind == 12:
                ans = str(sp.chebyshevu_root(u, v))
            elif kind == 13:
                ans = str(sp.hermite(u, x))
            elif kind == 14:
                ans = str(sp.laguerre(u, x))
            elif kind == 15:
                ans = str(sp.assoc_laguerre(u, v, x))
            q.put(ans)
        thread = threading.Thread(target=callback, args=(q, v, u, w, kind))
        thread.start()
        thread.join()
        return q.get()


    @staticmethod
    def transforms(f = '', a = 0, b = 0, L = [], kind = 1):
        q = queue.Queue()
        def callback(q, f = '', a = 0, b = 0, L = [], kind = 1):
            ans = ''
            if kind == 1:
                ans = str(sp.fft(L))  # Discrete Fourier Transform
            elif kind == 2:
                ans = str(sp.ifft(L))  # Inverse Discrete Fourier Transform
            elif kind == 3:
                ans = str(sp.ntt(L, prime=3 * 2 ** 8 + 1))  # Performs the Number Theoretic Transform (NTT)
            elif kind == 4:
                ans = str(sp.intt(L, prime=3 * 2 ** 8 + 1))  # Performs the Inverse Number Theoretic Transform (NTT)
            elif kind == 5:
                ans = str(sp.fwht(
                    L))  # Performs the Walsh Hadamard Transform (WHT), and uses Hadamard ordering for the sequence.
            elif kind == 6:
                ans = str(sp.ifwht(
                    L))  # Performs the Inverse Walsh Hadamard Transform (WHT), and uses Hadamard ordering for the sequence.
            elif kind == 7:
                ans = functions.math_display(str(sp.mellin_transform(f, x, s)))  # Mellin Transform
            elif kind == 8:
                ans = functions.math_display(
                    str(sp.inverse_mellin_transform(f, s, x, (a, b))))  # Inverse Mellin Transform
            elif kind == 9:
                from sympy import sin, cos, tan, exp, gamma, sinh, cosh, tanh
                ans = functions.math_display(str(functions.Laplace_Transform(f)))  # Laplace Transform
            elif kind == 10:
                ans = functions.math_display(str(sp.inverse_laplace_transform(f, s, t)))  # Inverse Laplace Transform
            elif kind == 11:
                ans = functions.math_display(str(functions.Fourier_Transform(f, a, b)))  # Fourier Transform
            elif kind == 12:
                ans = functions.math_display(str(sp.inverse_fourier_transform(f, w, x)))  # Invese Fourier Transform
            elif kind == 13:
                ans = functions.math_display(str(sp.sine_transform(f, x, w)))  # Fourier Sine Transform
            elif kind == 14:
                ans = functions.math_display(str(sp.inverse_sine_transform(f, w, x)))  # Inverse Fourier Sine Transform
            elif kind == 15:
                ans = functions.math_display(str(sp.cosine_transform(f, x, w)))  # Fourier Cosine Transform
            elif kind == 16:
                ans = functions.math_display(
                    str(sp.inverse_cosine_transform(f, w, x)))  # Inverse Fourier Cosine Transform
            elif kind == 17:
                ans = functions.math_display(str(sp.hankel_transform(f, r, w, 0)))  # Hankel Transform
            elif kind == 18:
                ans = functions.math_display(str(sp.inverse_hankel_transform(f, w, r, 0)))  # Inverse Hankel Transform
            q.put(ans)
        thread = threading.Thread(target=callback, args=(q, f , a , b , L , kind))
        thread.start()
        thread.join()
        return q.get()

    @staticmethod
    def bessel_function_1st(v):
        expr = (((-1)**e)*x**(2*e + v))/(2**(2*e + v)*sp.factorial(e)*sp.gamma(e + v + 1))
        return '(' + str(expr) + ')'

    @staticmethod
    def Legendre_Polynomials(n):
        f_x = ((x**2-1)**n)/(2**n*fact(n))
        return functions.math_display(str(sp.integrals.diff(f_x, x, n)))
    @staticmethod
    def Linear_First_Order(P, Q, R, x0='', y0='', kind=0):
        P = parse_expr(functions.tipa_parse_str(P))
        Q = parse_expr(functions.tipa_parse_str(Q))
        R = parse_expr(functions.tipa_parse_str(R))
        Q = '(' + R + ')' + '-' + '(' + Q + ')'
        y = sp.Function('y')
        if kind == 1:
            return str(sp.dsolve(sp.Eq(sp.integrals.diff(y(x), x) + P * y(x), Q), y(x), '1st_linear'))
        elif kind == 2:
            return str(sp.dsolve(sp.Eq(sp.integrals.diff(y(x), x) + P * y(x), Q), y(x), hint='Bernoulli'))
        elif kind == 3:
            return str(sp.dsolve(sp.Eq(P * sp.integrals.diff(y(x), x), Q), y(x), hint='separable', simplify=False))
        elif kind == 4:
            return str(sp.dsolve(sp.Eq(P * sp.integrals.diff(y(x), x), Q), y(x), hint='1st_homogeneous_coeff_best',
                                 simplify=False))
        elif kind == 5:
            return str(sp.dsolve(sp.Eq(P * sp.integrals.diff(y(x), x), Q), y(x), hint='1st_exact'))

    @staticmethod
    def Linear_Second_Order(P = '', Q = '', R = '', S = '', kind = 1):
        P = parse_expr(functions.tipa_parse_str(P))
        Q = parse_expr(functions.tipa_parse_str(Q))
        R = parse_expr(functions.tipa_parse_str(R))
        S = parse_expr(functions.tipa_parse_str(S))
        y = sp.Function('y')
        if kind == 1:
            eq = sp.Eq(P * sp.integrals.diff(y(x), x, 2) + Q * sp.integrals.diff(y(x), x) + R * y(x), S)
            return str(sp.dsolve(eq, hint='nth_linear_constant_coeff_homogeneous'))
        elif kind == 2:
            eq = sp.Eq(P * sp.integrals.diff(y(x), x, 2) + Q * sp.integrals.diff(y(x), x) + R * y(x), S)
            return str(sp.dsolve(eq, hint='nth_linear_constant_coeff_variation_of_parameters'))
        elif kind == 3:
            eq = sp.Eq(P * sp.integrals.diff(y(x), x, 2) + Q * sp.integrals.diff(y(x), x) + R * y(x), S)
            return str(sp.dsolve(eq, hint='nth_linear_constant_coeff_undetermined_coefficients'))
        elif kind == 4:
            eq = sp.Eq(P * sp.integrals.diff(y(x), x, 2) + Q * sp.integrals.diff(y(x), x) + R * y(x), S)
            return str(sp.dsolve(eq, hint='2nd_power_series_ordinary'))
        elif kind == 5:
            eq = sp.Eq(sp.integrals.diff(y(x), x, 2) + Q * sp.integrals.diff(y(x), x) + P * sp.integrals.diff(y(x),
                                                                                                              x) ** 2 + R * y(
                x), S)
            return str(sp.dsolve(eq, hint='Liouville'))
        elif kind == 6:
            eq = sp.Eq(P * sp.integrals.diff(y(x), x, 2) + Q * sp.integrals.diff(y(x), x) + R * y(x), S)
            return str(sp.dsolve(eq, hint='nth_linear_euler_eq_homogeneous'))
        elif kind == 7:
            eq = sp.Eq(P * sp.integrals.diff(y(x), x, 2) + Q * sp.integrals.diff(y(x), x) + R * y(x), S)
            return str(sp.dsolve(eq, hint='nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients').expand())
        elif kind == 8:
            eq = sp.Eq(P * sp.integrals.diff(y(x), x, 2) + Q * sp.integrals.diff(y(x), x) + R * y(x), S)
            return str(sp.dsolve(eq, hint='nth_linear_euler_eq_nonhomogeneous_variation_of_parameters').expand())

    @staticmethod
    def Partial_differential_equations(A = '0', B = '0', C = '0', D = '0', E = '0', F = '0', G = '0', kind = 1):
        A = parse_expr(functions.tipa_parse_str(A))
        B = parse_expr(functions.tipa_parse_str(B))
        C = parse_expr(functions.tipa_parse_str(C))
        D = parse_expr(functions.tipa_parse_str(D))
        E = parse_expr(functions.tipa_parse_str(E))
        F = parse_expr(functions.tipa_parse_str(F))
        G = parse_expr(functions.tipa_parse_str(G))
        X, Y, u, f = map(sp.Function, 'XYuf')
        uxx = sp.integrals.diff(u(x, y), x, 2)
        uyy = sp.integrals.diff(u(x, y), y, 2)
        uxy = sp.integrals.diff(u(x, y), x, y)
        ux = sp.integrals.diff(u(x, y), x)
        uy = sp.integrals.diff(u(x, y), x)
        fx = sp.integrals.diff(f(x, y), x)
        fy = sp.integrals.diff(f(x, y), x)
        if kind == 1:
            eq = sp.Eq(D * fx + E * fy + F * f(x, y), G)
            return str(sp.pdsolve(eq))
        elif kind == 2:
            eq = sp.Eq(D * fx + E * fy + F * f(x, y), G)
            return str(sp.pdsolve(eq))
        elif kind == 3:
            eq = sp.Eq(D * fx + E * fy + F * f(x, y), G)
            return str(sp.pdsolve(eq))
        elif kind == 4:
            eq = sp.Eq(A * uxx + B * uyy + C * uxy + D * ux + E * uy + F * u(x, y), G)
            return str(sp.pde_separate(eq, u(x, y), [X(x), Y(y)], strategy='mul'))
        elif kind == 5:
            eq = sp.Eq(A * uxx + B * uyy + C * uxy + D * ux + E * uy + F * u(x, y), G)
            return str(sp.pde_separate(eq, u(x, y), [X(x), Y(y)], strategy='add'))

def Sin(angle):
    if angle==30:
        return 0.5
    elif angle==90:
        return 1
    elif angle==180:
        return 0
    elif angle==0:
        return 0
    elif angle==360:
        return 0
    else:
        x = radians(angle)
        total = sin(x)
        return total

def Cos(angle):
    if angle==60:
        return 0.5
    elif angle==90:
        return 0
    elif angle==180:
        return -1
    elif angle==0:
        return 1
    elif angle==360:
        return 1
    elif angle==270:
        return 0
    else:
        x = radians(angle)
        total = cos(x)
        return total

def Tan(angle):
    if angle==45:
        return 1
    elif angle==180:
        return 0
    elif angle==0:
        return 0
    elif angle==360:
        return 0
    else:
        x = radians(angle)
        return tan(x) 

def aSin(x):
    if x==0.5:
        return 30
    elif x==1:
        return 90
    elif x == 2/sqrt(3):
        return 60
    elif x == 1/sqrt(2):
        return 45
    
    else:
        ans = asin(x)
        angle = degrees(ans)
        return angle

def aCos(x):
    if x==0.5:
        return 60
    elif x==1:
        return 0
    elif x == 2/sqrt(3):
        return 30
    elif x == 1/sqrt(2):
        return 45
    else:
        total = acos(x)
        angle = degrees(total)
        return angle

def aTan(x):
    if x==1:
        return 45
    elif x == 1/sqrt(3):
        return 30
    elif x == sqrt(3):
        return 60
    else:
        total = atan(x)
        angle = degrees(total)
        return angle
  
def fact(n):
    """ Takes one argument n which is an integer and returns the factorial of that integer value """
    if n==0:
        return 1
    else:
        return n*fact(n-1)

def P(n, r):
    """ Evaluates the permutation of integers n and r
            n should be greater than r"""
    return (fact(n)/fact(n-r))

def C(n, r):
    """ Evaluates the Combination of integers n and r  """
    return (fact(n)/(fact(n-r) * fact(r)))

    
def p(x,n):
    """ This function evaluates a complex number raised to the power n
            it takes on two arguments x and n.
            x: This is normally a complex number e.g. x = 2+3j
            n: is a numerical value which can be an integer or a float
            It returns a complex number"""
    angle = atan(x.imag/x.real)
    mod=abs(x)
    angle=n*angle
    mod =mod**n
    r = functions.float_analyzer(mod * cos(angle))
    i = functions.float_analyzer(mod * sin(angle))
    return '{:s}{:s}{:s}j'.format(r, '+' if eval(i)>0 else '-', str(abs(eval(i))))
def arg(x):
    """ It accepts one argument in complex form
        it then evaluates the argument of that complex number number
        It returns the angle in radians or degrees depending on the state of the variable color..
        """
    angle = atan(x.imag/x.real)
    return angle
def Arg(x):
    """ It accepts one argument in complex form
        it then evaluates the argument of that complex number number
        It returns the angle in radians or degrees depending on the state of the variable color..
        """
    angle = atan(x.imag/x.real)
    return degrees(angle)

class chattroom:
    """
        This class contains the functions used to perform messaging.
        The Chattroom is Database based, which it access remotely.

    """
    def __init__(self, fname, lname):
        self.fname = fname
        self.lname = lname

    def connect(self):
        try:
            conn = pymysql.connect(host='sql2.freemysqlhosting.net', user='sql2256164',
                               password='jH6*xS6%', db='sql2256164', port=3306, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor )
        except:
            return "Server can not be reach, please check your Internet Connection"
    def Login(self, username, password):
        self.username = username
        self.password = password
        

    
    


        
    
    
    
        
    
    


    



