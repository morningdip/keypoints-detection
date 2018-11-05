from __future__ import print_function
import random as rd

STYLE = {
    'fore':
    {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'purple': 35,
        'cyan': 36,
        'white': 37,
    },

    'back':
    {
        'black': 40,
        'red': 41,
        'green': 42,
        'yellow': 43,
        'blue': 44,
        'purple': 45,
        'cyan': 46,
        'white': 47,
    },

    'mode':
    {
        'normal': 0,
        'bold': 1,
        'underline': 4,
        'blink': 5,
        'invert': 7,
        'hide': 8,
    },

    'default':
    {
        'end': 0,
    }
}


class TerminalColor(object):
    def __init__(self):
        self.NOCOLOR = '\033[0m'
        self.RED = '\033[01;31m'
        self.GREEN = '\033[01;32m'
        self.YELLOW = '\033[01;33m'
        self.BLUE = '\033[01;34m'
        self.MAGENTA = '\033[01;35m'
        self.CYAN = '\033[01;36m'
        self.WHITE = '\033[01;37m'
        self.color = {
            'RED': self.RED, 'R': self.RED,
            'GREEN': self.GREEN, 'G': self.GREEN,
            'YELLOW': self.YELLOW, 'Y': self.YELLOW,
            'BLUE': self.BLUE, 'B': self.BLUE,
            'MAGENTA': self.MAGENTA, 'M': self.MAGENTA,
            'CYAN': self.CYAN, 'C': self.CYAN,
            'WHITE': self.WHITE, 'W': self.WHITE,
        }

    def colorformat(self, string, mode='', fore='', back=''):
        try:
            mode = '%s' % STYLE['mode'][mode] if mode else ''
            fore = '%s' % STYLE['fore'][fore] if fore else ''
            back = '%s' % STYLE['back'][back] if back else ''
            style = ';'.join([s for s in [mode, fore, back] if s])
            style = '\033[%sm' % style if style else ''
            end = '\033[%sm' % STYLE['default']['end'] if style else ''

            return '%s%s%s' % (style, string, end)
        except KeyError as e:
            print(self.colorformat('Format error, try:', mode='invert', back='red'))
            self.help()
            return string

    def printc(self, *args):
        try:
            print(''.join((self.color[args[0].upper()], ' '.join(
                (str(args[i]) for i in range(1, len(args)))))), self.NOCOLOR)
        except:
            self.printc('R', 'printmc format error, try:')
            self.help()

    def printmc(self, *args):
        try:
            lim = min(len(args) - 1, len(args[0])) - 1
            print(' '.join((self.color[args[0][min((i - 1, lim))].upper()] + str(args[i]) for i in range(1, len(args)))
                           ), self.NOCOLOR)
        except:
            self.printc('R', 'printmc format error, try:')
            self.help()

    def printa(self, *args):
        lim = len(args)
        co = ('WHITE',)
        color_list = ['R', 'G', 'Y', 'B', 'M', 'C']
        rd.shuffle(color_list)
        for i in range(1, lim):
            co = co + (color_list[i % 6],)
        tmpl = []
        tmpl.append(co)
        tmpl.extend(list(args))
        self.printmc(*tmpl)

    def clist(self):
        self.printmc(('WHITE', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE'), 'color:', 'RED',
                     'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE')

    def help(self):
        print(self.colorformat('colorformat', fore='cyan')
              + '('
              + self.colorformat('string, mode, fore, back', fore='yellow')
              + ')')

        print(self.colorformat('color (fore, back):\tblack, red, green, yellow, blue, purple, cyan, white', mode='invert'))
        print(self.colorformat('mode:\t\t\tnormal, bold, underline, invert', mode='invert'))
